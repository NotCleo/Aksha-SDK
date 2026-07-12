#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

// Set to 1 to silence ALL console output for timing runs (printf over the
// serial console costs a syscall + UART time per line). Set back to 0 to
// restore the full diagnostic log. Error paths via perror() still print.
#define QUIET 1
#if QUIET
#define printf(...) ((void)0)
#define fflush(x)   ((void)0)
#endif

// Hardware/timing report - goes to stderr so it prints even in QUIET builds.
#include <sys/time.h>
double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
double g_t0;   // program start, for the end-to-end total in the [TIME] report

// ==========================================
// 1. HARDWARE MAPPING
// ==========================================
#define SPI_PHYS_BASE   0x10115000UL
#define I2C_PHYS_BASE   0x10116000UL
#define MAP_SIZE        4096UL
#define MAP_MASK        (MAP_SIZE - 1)

volatile uint32_t *spi_base_virt = NULL;
volatile uint32_t *i2c_base_virt = NULL;

#define SPI_REG(offset)  (*(volatile uint32_t *)((uint8_t *)spi_base_virt + (offset)))
#define I2C_REG(offset)  (*(volatile uint32_t *)((uint8_t *)i2c_base_virt + (offset)))

// ==========================================
// 2. CONSTANTS
// ==========================================
#define ARDUCHIP_TEST1       0x00
#define ARDUCHIP_FRAMES       0x01   // 5MP-Plus shields only: bit[2:0] = frame count (0 = 1 frame)
#define ARDUCHIP_TIM          0x03   // 5MP-Plus shields only: timing/polarity control
#define VSYNC_LEVEL_MASK      0x02   // ARDUCHIP_TIM bit1: 0 = VSYNC high-active, 1 = low-active
#define ARDUCHIP_FIFO        0x04
#define FIFO_CLEAR_MASK       0x01
#define FIFO_START_MASK       0x02
#define FIFO_RDPTR_RST_MASK   0x10
#define FIFO_WRPTR_RST_MASK   0x20
#define ARDUCHIP_GPIO         0x06   // Experimental: some shield designs gate sensor RESET/PWDN here
#define GPIO_RESET_MASK        0x01  // 0 = sensor held in reset,  1 = normal operation
#define GPIO_PWDN_MASK          0x02  // 0 = normal operation,      1 = sensor in standby/powerdown
#define GPIO_PWREN_MASK          0x04  // 0 = onboard LDO disabled,  1 = onboard LDO enabled
#define ARDUCHIP_REV         0x40   // Shield hardware revision (read-only)
#define VER_LOW_MASK          0x3F
#define VER_HIGH_MASK          0xC0
#define ARDUCHIP_TRIG        0x41
#define CAP_DONE_MASK        0x08
#define ARDUCHIP_FIFO_SIZE1  0x42
#define ARDUCHIP_FIFO_SIZE2  0x43
#define ARDUCHIP_FIFO_SIZE3  0x44
#define BURST_FIFO_READ      0x3C

// OV5642 uses 16-bit register addressing and a different I2C slave address
// than the OV2640 (7-bit 0x3C -> 8-bit write address 0x78).
#define OV5642_ADDR           0x3C
#define OV5642_CHIPID_HIGH    0x300A
#define OV5642_CHIPID_LOW     0x300B

// ==========================================
// 3. UTILITIES
// ==========================================
void *map_physical_memory(uint32_t phys_addr) {
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd == -1) { perror("Error opening /dev/mem"); exit(EXIT_FAILURE); }
    void *mapped_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, phys_addr & ~MAP_MASK);
    if (mapped_base == MAP_FAILED) { perror("Error mapping memory"); close(mem_fd); exit(EXIT_FAILURE); }
    close(mem_fd);
    return (void *)((uint8_t *)mapped_base + (phys_addr & MAP_MASK));
}

// ==========================================
// 4. LOW-LEVEL DRIVERS (BLIND MODE)
// ==========================================
// SPI side talks to the ArduCAM shield's own FIFO/SPI-bridge chip. Same
// shield protocol as the OV2640 driver - unchanged.

// Every register access to the AXI peripherals via /dev/mem costs ~8us on
// this SoC (measured: TRIG fast-polls at ~100us each = ~12 accesses). That,
// not the SPI clock, bounds throughput - so the fast path below minimizes
// register accesses per byte.
uint32_t spi_fifo_depth = 1;   // learned in spi_init(); 1 = core has no FIFOs

uint8_t spi_transfer(uint8_t data);

void spi_init() {
    SPI_REG(0x40) = 0x0A;
    for (int i = 0; i < 8; i++) (void)SPI_REG(0x64);  // ~25us reset settle, no 10ms usleep tick
    SPI_REG(0x60) = 0x186;      // master, manual SS, transactions inhibited
    SPI_REG(0x70) = 0xFFFFFFFF;
    // Probe the TX FIFO depth once: while inhibited, bytes queue in the TX
    // FIFO; count how many fit before TX_FULL (SPISR bit3) asserts. A core
    // synthesized without FIFOs reports full after a single byte.
    uint32_t d = 0;
    while (d < 256 && !(SPI_REG(0x64) & 0x08)) { SPI_REG(0x68) = 0x00; d++; }
    spi_fifo_depth = d ? d : 1;
    printf("  (SPI core FIFO depth: %u)\n", spi_fifo_depth);
    // Reset to discard the probe bytes, then run with the transaction
    // inhibit cleared permanently - the old code cleared it again on every
    // byte with a read-modify-write, two wasted accesses per transfer.
    SPI_REG(0x40) = 0x0A;
    for (int i = 0; i < 8; i++) (void)SPI_REG(0x64);  // ~25us reset settle, no 10ms usleep tick
    SPI_REG(0x60) = 0x086;      // same config, inhibit clear
    SPI_REG(0x70) = 0xFFFFFFFF;

    // One-time hardware characterization. CS stays deasserted, so the 500
    // test transfers clock the bus but the shield ignores them. This is the
    // ground truth for "is the fabric config the bottleneck": reg-read cost
    // is the /dev/mem AXI access latency, byte cost implies the actual SCK.
    double t0 = now_ms();
    for (int i = 0; i < 2000; i++) (void)SPI_REG(0x64);
    double t1 = now_ms();
    for (int i = 0; i < 500; i++) (void)spi_transfer(0x00);
    double t2 = now_ms();
    double rd_us = (t1 - t0) * 1000.0 / 2000.0;
    double byte_us = (t2 - t1) * 1000.0 / 500.0;
    // NOTE: the per-byte cost here is dominated by /dev/mem AXI register
    // access latency (~3 accesses per byte), NOT by the SPI clock. SCK is
    // fixed in the bitstream at ext_spi_clk/C_SCK_RATIO = 40MHz/16 = 2.5MHz;
    // no DTS or software setting can change it. (An earlier build printed a
    // bogus "SCK approx" figure derived from this number - ignore old logs.)
    fprintf(stderr, "[HW] SPI FIFO depth=%u | reg-read=%.2fus | byte=%.1fus\n",
            spi_fifo_depth, rd_us, byte_us);
}

uint8_t spi_transfer(uint8_t data) {
    SPI_REG(0x68) = data;

    // Safety timeout loop for SPI
    int timeout = 0;
    while((SPI_REG(0x64) & 0x01) != 0) {
        if(++timeout > 100000) break; // Break if stuck
    }
    return (uint8_t)SPI_REG(0x6C);
}

// Fast bulk read for use inside an already-CS-asserted burst, INTERLEAVED:
// keep the TX FIFO topped up while draining RX, so the SPI wire never sits
// idle waiting for the CPU (the old write-16/wait/read-16 pattern stalled
// the wire during every drain - measured 7.3us/byte; this lands near the
// ~2.25-accesses-per-byte PIO floor). Correctness is flow-controlled, not
// timed: the software in-flight counter (sent - rcvd <= FIFO depth) is the
// RX-overflow guard - even if every in-flight byte lands in RX before we
// drain, RX holds at most `depth` bytes. RX status is only trusted on the
// positive event (SPISR bit0 = not-empty, then OCY = count-1), never on an
// idle-state predicate, so stale status reads cannot fake completion.
// Byte order/count identical to the one-at-a-time path; falls back to it
// if the core has no FIFOs.
void spi_read_burst(uint8_t *dst, uint32_t n) {
    if (spi_fifo_depth < 2) {
        for (uint32_t i = 0; i < n; i++) dst[i] = spi_transfer(0x00);
        return;
    }
    uint32_t sent = 0, rcvd = 0;
    const uint32_t depth = spi_fifo_depth;
    int idle = 0;
    while (rcvd < n) {
        // Top off TX: bounded by the in-flight cap, never by FIFO status.
        while (sent < n && (sent - rcvd) < depth) {
            SPI_REG(0x68) = 0x00;
            sent++;
        }
        // Drain whatever RX already holds: one status read + one occupancy
        // read amortized over the whole batch, then blind DRR reads. More
        // bytes may arrive while draining - picked up next iteration.
        if (!(SPI_REG(0x64) & 0x01)) {                    // RX not empty
            uint32_t avail = (SPI_REG(0x78) & 0xFF) + 1;  // OCY reads count-1
            uint32_t inflight = sent - rcvd;
            if (avail > inflight) avail = inflight;       // paranoia clamp
            while (avail--) dst[rcvd++] = (uint8_t)SPI_REG(0x6C);
            idle = 0;
        } else if (++idle > 200000) {
            break;   // bounded, like the old per-chunk timeout - never hang
        }
    }
}

// No inter-transaction usleep: the ArduChip needs none (the Arduino driver
// runs back-to-back at 8MHz), and usleep(100) rounds up to a ~10ms
// scheduler tick on this kernel.
void arducam_write_reg(uint8_t addr, uint8_t data) {
    SPI_REG(0x70) = 0xFFFFFFFE;
    spi_transfer(addr | 0x80);
    spi_transfer(data);
    SPI_REG(0x70) = 0xFFFFFFFF;
}

uint8_t arducam_read_reg(uint8_t addr) {
    SPI_REG(0x70) = 0xFFFFFFFE;
    spi_transfer(addr & 0x7F);
    uint8_t val = spi_transfer(0x00);
    SPI_REG(0x70) = 0xFFFFFFFF;
    return val;
}

// --- I2C (BLIND MODE - NO CHECKS ON WRITE) ---
void i2c_init() {
    I2C_REG(0x40) = 0x0A;
    for (int i = 0; i < 8; i++) (void)I2C_REG(0x104); // ~25us reset settle, no 10ms usleep tick
    I2C_REG(0x100) = 0x01;
}

// OV5642 registers are addressed with 16 bits (high byte, then low byte),
// unlike the OV2640's single 8-bit register address. Same blind AXI-IIC
// TX_FIFO framing (START=0x100 / STOP=0x200 control bits on offset 0x108)
// as the OV2640 driver, just with one extra address byte in the sequence.
void i2c_write_reg16(uint16_t reg, uint8_t val) {
    I2C_REG(0x40) = 0x0A;
    for (int i = 0; i < 32; i++) (void)I2C_REG(0x104); // reset settle (32 reads: optimized builds cut per-read time in half)
    I2C_REG(0x100) = 0x01;

    I2C_REG(0x108) = 0x100 | (OV5642_ADDR << 1);  // START + slave write address
    I2C_REG(0x108) = (reg >> 8) & 0xFF;            // register address high byte
    I2C_REG(0x108) = reg & 0xFF;                   // register address low byte
    I2C_REG(0x108) = 0x200 | val;                  // data byte + STOP

    // TWO-PHASE completion wait (polling, not usleep - this kernel rounds
    // usleep up to a ~10ms tick). Phase 1: the transaction must first be
    // SEEN active (TX FIFO holding our bytes, or bus busy). "TX empty &&
    // bus idle" is ALSO the idle state: at -O2/-O3 a status read issued
    // back-to-back with the FIFO pushes can return a stale pre-push
    // snapshot, so a single-phase poll "completed" instantly and the next
    // call's soft reset killed the transaction mid-byte - every register
    // write silently lost while reads (which wait on a positive event)
    // stayed perfect. Do not simplify this back to one phase.
    int timeout = 0;
    for (;;) {
        uint32_t sr = I2C_REG(0x104);
        if (!(sr & 0x80) || (sr & 0x04)) break;   // bytes visible in TX FIFO, or bus already busy
        if (++timeout > 100000) break;
    }
    // Phase 2: now "TX drained && bus idle" genuinely means done. The real
    // transaction is ~450us at 100kHz SCL (fabric: axi_iic IIC_FREQ_KHZ=100).
    timeout = 0;
    for (;;) {
        uint32_t sr = I2C_REG(0x104);
        if ((sr & 0x80) && !(sr & 0x04)) break;   // TX_FIFO_EMPTY && !BUS_BUSY
        if (++timeout > 300000) break;             // ~1s cap, never hang
    }
}

// Needed (not just for diagnostics) because the OV5642 preview-mode init
// below does a read-modify-write on two registers. Uses the AXI IIC core's
// "dynamic mode" repeated-start read: write reg addr with no STOP, then a
// repeated START + read address, then STOP+count to pull one byte into
// RX_FIFO (0x10C). Bounded timeout on RX_FIFO_OCY (0x114) instead of a
// blind wait, since this path is less proven than the write path.
uint8_t i2c_read_reg16(uint16_t reg) {
    I2C_REG(0x40) = 0x0A;
    for (int i = 0; i < 32; i++) (void)I2C_REG(0x104);   // reset settle, matches write path
    I2C_REG(0x100) = 0x01;

    I2C_REG(0x108) = 0x100 | (OV5642_ADDR << 1);        // START + slave write addr
    I2C_REG(0x108) = (reg >> 8) & 0xFF;                  // reg addr high byte
    I2C_REG(0x108) = reg & 0xFF;                         // reg addr low byte (no STOP)

    I2C_REG(0x108) = 0x100 | (OV5642_ADDR << 1) | 0x01;  // repeated START + slave read addr
    I2C_REG(0x108) = 0x200 | 0x01;                        // STOP + read 1 byte

    // Poll SR RX_FIFO_EMPTY (bit6) until the byte lands. Also fixes an old
    // latent bug: this used to poll 0x114 (which is TX_FIFO occupancy, not
    // RX) and then blind-sleep, so a failed transaction could silently
    // return a STALE byte left in the RX FIFO from an earlier read.
    int timeout = 0;
    while (I2C_REG(0x104) & 0x40) {          // RX_FIFO_EMPTY
        if (++timeout > 300000) break;        // ~1s cap
    }
    return (uint8_t)(I2C_REG(0x10C) & 0xFF); // RX_FIFO
}

// ==========================================
// 5. CONFIGURATION ARRAY (OV5642 QVGA Preview base init)
// ==========================================
// Verbatim from ArduCAM's official OV5642 driver
// (https://github.com/ArduCAM/Arduino/blob/master/ArduCAM/ov5642_regs.h,
// OV5642_QVGA_Preview array). Self-contained software reset + PLL/analog/ISP
// bring-up + 320x240 output window. It leaves the sensor in a YUV-ish state,
// but per ArduCAM's own ArduCAM.cpp InitCAM(), actually landing on correct
// YUV422 preview output needs the extra registers applied after this array
// in main() below (0x501f/0x4300/etc. get overridden here to different
// values than the array leaves them at).
const uint16_t OV5642_QVGA_Preview[][2] = {
    {0x3103, 0x93}, {0x3008, 0x82}, {0x3017, 0x7f}, {0x3018, 0xfc}, {0x3810, 0xc2}, {0x3615, 0xf0},
    {0x3000, 0x00}, {0x3001, 0x00}, {0x3002, 0x5c}, {0x3003, 0x00}, {0x3004, 0xff}, {0x3005, 0xff},
    {0x3006, 0x43}, {0x3007, 0x37}, {0x3011, 0x08}, {0x3010, 0x10}, {0x460c, 0x22}, {0x3815, 0x04},
    {0x370c, 0xa0}, {0x3602, 0xfc}, {0x3612, 0xff}, {0x3634, 0xc0}, {0x3613, 0x00}, {0x3605, 0x7c},
    {0x3621, 0x09}, {0x3622, 0x60}, {0x3604, 0x40}, {0x3603, 0xa7}, {0x3603, 0x27}, {0x4000, 0x21},
    {0x401d, 0x22}, {0x3600, 0x54}, {0x3605, 0x04}, {0x3606, 0x3f}, {0x3c01, 0x80}, {0x5000, 0x4f},
    {0x5020, 0x04}, {0x5181, 0x79}, {0x5182, 0x00}, {0x5185, 0x22}, {0x5197, 0x01}, {0x5001, 0xff},
    {0x5500, 0x0a}, {0x5504, 0x00}, {0x5505, 0x7f}, {0x5080, 0x08}, {0x300e, 0x18}, {0x4610, 0x00},
    {0x471d, 0x05}, {0x4708, 0x06}, {0x3808, 0x02}, {0x3809, 0x80}, {0x380a, 0x01}, {0x380b, 0xe0},
    {0x380e, 0x07}, {0x380f, 0xd0}, {0x501f, 0x00}, {0x5000, 0x4f}, {0x4300, 0x30}, {0x3503, 0x07},
    {0x3501, 0x73}, {0x3502, 0x80}, {0x350b, 0x00}, {0x3503, 0x07}, {0x3824, 0x11}, {0x3501, 0x1e},
    {0x3502, 0x80}, {0x350b, 0x7f}, {0x380c, 0x0c}, {0x380d, 0x80}, {0x380e, 0x03}, {0x380f, 0xe8},
    {0x3a0d, 0x04}, {0x3a0e, 0x03}, {0x3818, 0xc1}, {0x3705, 0xdb}, {0x370a, 0x81}, {0x3801, 0x80},
    {0x3621, 0x87}, {0x3801, 0x50}, {0x3803, 0x08}, {0x3827, 0x08}, {0x3810, 0x40}, {0x3804, 0x05},
    {0x3805, 0x00}, {0x5682, 0x05}, {0x5683, 0x00}, {0x3806, 0x03}, {0x3807, 0xc0}, {0x5686, 0x03},
    {0x5687, 0xbc}, {0x3a00, 0x78}, {0x3a1a, 0x05}, {0x3a13, 0x30}, {0x3a18, 0x00}, {0x3a19, 0x7c},
    {0x3a08, 0x12}, {0x3a09, 0xc0}, {0x3a0a, 0x0f}, {0x3a0b, 0xa0}, {0x350c, 0x07}, {0x350d, 0xd0},
    {0x3500, 0x00}, {0x3501, 0x00}, {0x3502, 0x00}, {0x350a, 0x00}, {0x350b, 0x00}, {0x3503, 0x00},
    {0x528a, 0x02}, {0x528b, 0x04}, {0x528c, 0x08}, {0x528d, 0x08}, {0x528e, 0x08}, {0x528f, 0x10},
    {0x5290, 0x10}, {0x5292, 0x00}, {0x5293, 0x02}, {0x5294, 0x00}, {0x5295, 0x02}, {0x5296, 0x00},
    {0x5297, 0x02}, {0x5298, 0x00}, {0x5299, 0x02}, {0x529a, 0x00}, {0x529b, 0x02}, {0x529c, 0x00},
    {0x529d, 0x02}, {0x529e, 0x00}, {0x529f, 0x02}, {0x3030, 0x0b}, {0x3a02, 0x00}, {0x3a03, 0x7d},
    {0x3a04, 0x00}, {0x3a14, 0x00}, {0x3a15, 0x7d}, {0x3a16, 0x00}, {0x3a00, 0x78}, {0x3a08, 0x09},
    {0x3a09, 0x60}, {0x3a0a, 0x07}, {0x3a0b, 0xd0}, {0x3a0d, 0x08}, {0x3a0e, 0x06}, {0x5193, 0x70},
    {0x589b, 0x04}, {0x589a, 0xc5}, {0x401e, 0x20}, {0x4001, 0x42}, {0x401c, 0x04}, {0x528a, 0x01},
    {0x528b, 0x04}, {0x528c, 0x08}, {0x528d, 0x10}, {0x528e, 0x20}, {0x528f, 0x28}, {0x5290, 0x30},
    {0x5292, 0x00}, {0x5293, 0x01}, {0x5294, 0x00}, {0x5295, 0x04}, {0x5296, 0x00}, {0x5297, 0x08},
    {0x5298, 0x00}, {0x5299, 0x10}, {0x529a, 0x00}, {0x529b, 0x20}, {0x529c, 0x00}, {0x529d, 0x28},
    {0x529e, 0x00}, {0x529f, 0x30}, {0x5282, 0x00}, {0x5300, 0x00}, {0x5301, 0x20}, {0x5302, 0x00},
    {0x5303, 0x7c}, {0x530c, 0x00}, {0x530d, 0x0c}, {0x530e, 0x20}, {0x530f, 0x80}, {0x5310, 0x20},
    {0x5311, 0x80}, {0x5308, 0x20}, {0x5309, 0x40}, {0x5304, 0x00}, {0x5305, 0x30}, {0x5306, 0x00},
    {0x5307, 0x80}, {0x5314, 0x08}, {0x5315, 0x20}, {0x5319, 0x30}, {0x5316, 0x10}, {0x5317, 0x00},
    {0x5318, 0x02}, {0x5402, 0x3f}, {0x5403, 0x00}, {0x3406, 0x00}, {0x5180, 0xff}, {0x5181, 0x52},
    {0x5182, 0x11}, {0x5183, 0x14}, {0x5184, 0x25}, {0x5185, 0x24}, {0x5186, 0x06}, {0x5187, 0x08},
    {0x5188, 0x08}, {0x5189, 0x7c}, {0x518a, 0x60}, {0x518b, 0xb2}, {0x518c, 0xb2}, {0x518d, 0x44},
    {0x518e, 0x3d}, {0x518f, 0x58}, {0x5190, 0x46}, {0x5191, 0xf8}, {0x5192, 0x04}, {0x5193, 0x70},
    {0x5194, 0xf0}, {0x5195, 0xf0}, {0x5196, 0x03}, {0x5197, 0x01}, {0x5198, 0x04}, {0x5199, 0x12},
    {0x519a, 0x04}, {0x519b, 0x00}, {0x519c, 0x06}, {0x519d, 0x82}, {0x519e, 0x00}, {0x5025, 0x80},
    {0x5583, 0x40}, {0x5584, 0x40}, {0x5580, 0x02}, {0x5000, 0xcf}, {0x3710, 0x10}, {0x3632, 0x51},
    {0x3702, 0x10}, {0x3703, 0xb2}, {0x3704, 0x18}, {0x370b, 0x40}, {0x370d, 0x03}, {0x3631, 0x01},
    {0x3632, 0x52}, {0x3606, 0x24}, {0x3620, 0x96}, {0x5785, 0x07}, {0x3a13, 0x30}, {0x3600, 0x52},
    {0x3604, 0x48}, {0x3606, 0x1b}, {0x370d, 0x0b}, {0x370f, 0xc0}, {0x3709, 0x01}, {0x3823, 0x00},
    {0x5007, 0x00}, {0x5009, 0x00}, {0x5011, 0x00}, {0x5013, 0x00}, {0x519e, 0x00}, {0x5086, 0x00},
    {0x5087, 0x00}, {0x5088, 0x00}, {0x5089, 0x00}, {0x302b, 0x00}, {0x3808, 0x01}, {0x3809, 0x40},
    {0x380a, 0x00}, {0x380b, 0xf0}, {0x3a00, 0x78}, {0x5001, 0xFF}, {0x5583, 0x50}, {0x5584, 0x50},
    {0x5580, 0x02}, {0x3c01, 0x80}, {0x3c00, 0x04}, {0x5800, 0x48}, {0x5801, 0x31}, {0x5802, 0x21},
    {0x5803, 0x1b}, {0x5804, 0x1a}, {0x5805, 0x1e}, {0x5806, 0x29}, {0x5807, 0x38}, {0x5808, 0x26},
    {0x5809, 0x17}, {0x580a, 0x11}, {0x580e, 0x13}, {0x580f, 0x1a}, {0x5810, 0x15}, {0x5818, 0x11},
    {0x5820, 0x12}, {0x5828, 0x17}, {0x5830, 0x28}, {0x5831, 0x1a}, {0x5832, 0x11}, {0x5836, 0x15},
    {0x5837, 0x1d}, {0x5838, 0x6e}, {0x5839, 0x39}, {0x583a, 0x27}, {0x583b, 0x1f}, {0x583c, 0x1e},
    {0x583d, 0x23}, {0x583e, 0x2f}, {0x583f, 0x41}, {0x584e, 0x10}, {0x584f, 0x10}, {0x5850, 0x11},
    {0x5854, 0x10}, {0x5855, 0x10}, {0x5856, 0x10}, {0x5864, 0x17}, {0x5865, 0x14}, {0x5866, 0x18},
    {0x5867, 0x18}, {0x5868, 0x16}, {0x5869, 0x12}, {0x586a, 0x1b}, {0x586b, 0x1a}, {0x586c, 0x16},
    {0x586d, 0x16}, {0x586e, 0x18}, {0x586f, 0x1f}, {0x5870, 0x1c}, {0x5871, 0x16}, {0x5872, 0x10},
    {0x5874, 0x13}, {0x5875, 0x1c}, {0x5876, 0x1e}, {0x5877, 0x17}, {0x5878, 0x11}, {0x5879, 0x11},
    {0x587a, 0x14}, {0x587b, 0x1e}, {0x587c, 0x1c}, {0x587d, 0x1c}, {0x587e, 0x1a}, {0x587f, 0x1a},
    {0x5880, 0x1b}, {0x5881, 0x1f}, {0x5882, 0x14}, {0x5883, 0x1a}, {0x5884, 0x1d}, {0x5885, 0x1e},
    {0x5886, 0x1a}, {0x5887, 0x1a}, {0x5180, 0xff}, {0x5181, 0x52}, {0x5182, 0x11}, {0x5183, 0x14},
    {0x5184, 0x25}, {0x5185, 0x24}, {0x5186, 0x14}, {0x5187, 0x14}, {0x5188, 0x14}, {0x5189, 0x69},
    {0x518a, 0x60}, {0x518b, 0xa2}, {0x518c, 0x9c}, {0x518d, 0x36}, {0x518e, 0x34}, {0x518f, 0x54},
    {0x5190, 0x4c}, {0x5191, 0xf8}, {0x5192, 0x04}, {0x5193, 0x70}, {0x5194, 0xf0}, {0x5195, 0xf0},
    {0x5196, 0x03}, {0x5197, 0x01}, {0x5198, 0x05}, {0x5199, 0x2f}, {0x519a, 0x04}, {0x519b, 0x00},
    {0x519c, 0x06}, {0x519d, 0xa0}, {0x519e, 0xa0}, {0x528a, 0x00}, {0x528b, 0x01}, {0x528c, 0x04},
    {0x528d, 0x08}, {0x528e, 0x10}, {0x528f, 0x20}, {0x5290, 0x30}, {0x5292, 0x00}, {0x5293, 0x00},
    {0x5294, 0x00}, {0x5295, 0x01}, {0x5296, 0x00}, {0x5297, 0x04}, {0x5298, 0x00}, {0x5299, 0x08},
    {0x529a, 0x00}, {0x529b, 0x10}, {0x529c, 0x00}, {0x529d, 0x20}, {0x529e, 0x00}, {0x529f, 0x30},
    {0x5282, 0x00}, {0x5300, 0x00}, {0x5301, 0x20}, {0x5302, 0x00}, {0x5303, 0x7c}, {0x530c, 0x00},
    {0x530d, 0x10}, {0x530e, 0x20}, {0x530f, 0x80}, {0x5310, 0x20}, {0x5311, 0x80}, {0x5308, 0x20},
    {0x5309, 0x40}, {0x5304, 0x00}, {0x5305, 0x30}, {0x5306, 0x00}, {0x5307, 0x80}, {0x5314, 0x08},
    {0x5315, 0x20}, {0x5319, 0x30}, {0x5316, 0x10}, {0x5317, 0x00}, {0x5318, 0x02}, {0x5380, 0x01},
    {0x5381, 0x00}, {0x5382, 0x00}, {0x5383, 0x1f}, {0x5384, 0x00}, {0x5385, 0x06}, {0x5386, 0x00},
    {0x5387, 0x00}, {0x5388, 0x00}, {0x5389, 0xE1}, {0x538A, 0x00}, {0x538B, 0x2B}, {0x538C, 0x00},
    {0x538D, 0x00}, {0x538E, 0x00}, {0x538F, 0x10}, {0x5390, 0x00}, {0x5391, 0xB3}, {0x5392, 0x00},
    {0x5393, 0xA6}, {0x5394, 0x08}, {0x5480, 0x0c}, {0x5481, 0x18}, {0x5482, 0x2f}, {0x5483, 0x55},
    {0x5484, 0x64}, {0x5485, 0x71}, {0x5486, 0x7d}, {0x5487, 0x87}, {0x5488, 0x91}, {0x5489, 0x9a},
    {0x548A, 0xaa}, {0x548B, 0xb8}, {0x548C, 0xcd}, {0x548D, 0xdd}, {0x548E, 0xea}, {0x548F, 0x1d},
    {0x5490, 0x05}, {0x5491, 0x00}, {0x5492, 0x04}, {0x5493, 0x20}, {0x5494, 0x03}, {0x5495, 0x60},
    {0x5496, 0x02}, {0x5497, 0xB8}, {0x5498, 0x02}, {0x5499, 0x86}, {0x549A, 0x02}, {0x549B, 0x5B},
    {0x549C, 0x02}, {0x549D, 0x3B}, {0x549E, 0x02}, {0x549F, 0x1C}, {0x54A0, 0x02}, {0x54A1, 0x04},
    {0x54A2, 0x01}, {0x54A3, 0xED}, {0x54A4, 0x01}, {0x54A5, 0xC5}, {0x54A6, 0x01}, {0x54A7, 0xA5},
    {0x54A8, 0x01}, {0x54A9, 0x6C}, {0x54AA, 0x01}, {0x54AB, 0x41}, {0x54AC, 0x01}, {0x54AD, 0x20},
    {0x54AE, 0x00}, {0x54AF, 0x16}, {0x54B0, 0x01}, {0x54B1, 0x20}, {0x54B2, 0x00}, {0x54B3, 0x10},
    {0x54B4, 0x00}, {0x54B5, 0xf0}, {0x54B6, 0x00}, {0x54B7, 0xDF}, {0x5402, 0x3f}, {0x5403, 0x00},
    {0x5500, 0x10}, {0x5502, 0x00}, {0x5503, 0x06}, {0x5504, 0x00}, {0x5505, 0x7f}, {0x5025, 0x80},
    {0x3a0f, 0x30}, {0x3a10, 0x28}, {0x3a1b, 0x30}, {0x3a1e, 0x28}, {0x3a11, 0x61}, {0x3a1f, 0x10},
    {0x5688, 0xfd}, {0x5689, 0xdf}, {0x568a, 0xfe}, {0x568b, 0xef}, {0x568c, 0xfe}, {0x568d, 0xef},
    {0x568e, 0xaa}, {0x568f, 0xaa},
    {0xffff, 0xff}
};

// ==========================================
// 5b. JPEG-MODE CONFIGURATION ARRAYS
// ==========================================
// Verbatim from the same ov5642_regs.h (comments stripped). ArduCAM's
// InitCAM() JPEG flow for the Mini-5MP-Plus is: QVGA_Preview ->
// JPEG_Capture_QSXGA -> ov5642_320x240 -> fixup regs -> chosen size array
// (OV5642_set_JPEG_size). The sensor's own ISP + JPEG encoder does the
// compression; the shield FIFO just stores the byte stream (8MB max).

const uint16_t OV5642_JPEG_Capture_QSXGA[][2] = {
    {0x3503, 0x07}, {0x3000, 0x00}, {0x3001, 0x00}, {0x3002, 0x00}, {0x3003, 0x00}, {0x3005, 0xff},
    {0x3006, 0xff}, {0x3007, 0x3f}, {0x350c, 0x07}, {0x350d, 0xd0}, {0x3602, 0xe4}, {0x3612, 0xac},
    {0x3613, 0x44}, {0x3621, 0x27}, {0x3622, 0x08}, {0x3623, 0x22}, {0x3604, 0x60}, {0x3705, 0xda},
    {0x370a, 0x80}, {0x3801, 0x8a}, {0x3803, 0x0a}, {0x3804, 0x0a}, {0x3805, 0x20}, {0x3806, 0x07},
    {0x3807, 0x98}, {0x3808, 0x0a}, {0x3809, 0x20}, {0x380a, 0x07}, {0x380b, 0x98}, {0x380c, 0x0c},
    {0x380d, 0x80}, {0x380e, 0x07}, {0x380f, 0xd0}, {0x3810, 0xc2}, {0x3815, 0x44}, {0x3818, 0xc8},
    {0x3824, 0x01}, {0x3827, 0x0a}, {0x3a00, 0x78}, {0x3a0d, 0x10}, {0x3a0e, 0x0d}, {0x3a10, 0x32},
    {0x3a1b, 0x3c}, {0x3a1e, 0x32}, {0x3a11, 0x80}, {0x3a1f, 0x20}, {0x3a00, 0x78}, {0x460b, 0x35},
    {0x471d, 0x00}, {0x4713, 0x03}, {0x471c, 0x50}, {0x5682, 0x0a}, {0x5683, 0x20}, {0x5686, 0x07},
    {0x5687, 0x98}, {0x5001, 0x4f}, {0x589b, 0x00}, {0x589a, 0xc0}, {0x4407, 0x08}, {0x589b, 0x00},
    {0x589a, 0xc0}, {0x3002, 0x0c}, {0x3002, 0x00}, {0x3503, 0x00}, {0x5025, 0x80}, {0x3a0f, 0x48},
    {0x3a10, 0x40}, {0x3a1b, 0x4a}, {0x3a1e, 0x3e}, {0x3a11, 0x70}, {0x3a1f, 0x20},
    {0xffff, 0xff},
};

const uint16_t ov5642_320x240[][2] = {
    {0x3800, 0x1}, {0x3801, 0xa8}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0x1}, {0x3809, 0x40}, {0x380a, 0x0}, {0x380b, 0xf0},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98}, {0x3801, 0xb0},
    {0xffff, 0xff},
};

const uint16_t ov5642_640x480[][2] = {
    {0x3800, 0x1}, {0x3801, 0xa8}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0x2}, {0x3809, 0x80}, {0x380a, 0x1}, {0x380b, 0xe0},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98}, {0x3801, 0xb0},
    {0xffff, 0xff},
};

const uint16_t ov5642_1024x768[][2] = {
    {0x3800, 0x1}, {0x3801, 0xb0}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0x4}, {0x3809, 0x0}, {0x380a, 0x3}, {0x380b, 0x0},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98},
    {0xffff, 0xff},
};

const uint16_t ov5642_1280x960[][2] = {
    {0x3800, 0x1}, {0x3801, 0xb0}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0x5}, {0x3809, 0x00}, {0x380a, 0x3}, {0x380b, 0xc0},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98},
    {0xffff, 0xff},
};

const uint16_t ov5642_1600x1200[][2] = {
    {0x3800, 0x1}, {0x3801, 0xb0}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0x6}, {0x3809, 0x40}, {0x380a, 0x4}, {0x380b, 0xb0},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98},
    {0xffff, 0xff},
};

const uint16_t ov5642_2048x1536[][2] = {
    {0x3800, 0x01}, {0x3801, 0xb0}, {0x3802, 0x00}, {0x3803, 0x0a}, {0x3804, 0x0a}, {0x3805, 0x20},
    {0x3806, 0x07}, {0x3807, 0x98}, {0x3808, 0x08}, {0x3809, 0x00}, {0x380a, 0x06}, {0x380b, 0x00},
    {0x380c, 0x0c}, {0x380d, 0x80}, {0x380e, 0x07}, {0x380f, 0xd0}, {0x3810, 0xc2}, {0x3815, 0x44},
    {0x3818, 0xa8}, {0x3824, 0x01}, {0x3827, 0x0a}, {0x3a00, 0x78}, {0x3a0d, 0x10}, {0x3a0e, 0x0d},
    {0x3a00, 0x78}, {0x460b, 0x35}, {0x471d, 0x00}, {0x471c, 0x50}, {0x5682, 0x0a}, {0x5683, 0x20},
    {0x5686, 0x07}, {0x5687, 0x98}, {0x589b, 0x00}, {0x589a, 0xc0}, {0x589b, 0x00}, {0x589a, 0xc0},
    {0x3002, 0x0c}, {0x3002, 0x00}, {0x4300, 0x32}, {0x460b, 0x35}, {0x3002, 0x0c}, {0x3002, 0x00},
    {0x4713, 0x02}, {0x4600, 0x80}, {0x4721, 0x02}, {0x471c, 0x40}, {0x4408, 0x00}, {0x460c, 0x22},
    {0x3815, 0x04}, {0x3818, 0xc8}, {0x501f, 0x00}, {0x5002, 0xe0}, {0x440a, 0x01}, {0x4402, 0x90},
    {0x3811, 0xf0}, {0x3818, 0xa8}, {0x3621, 0x10},
    {0xffff, 0xff},
};

const uint16_t ov5642_2592x1944[][2] = {
    {0x3800, 0x1}, {0x3801, 0xb0}, {0x3802, 0x0}, {0x3803, 0xa}, {0x3804, 0xa}, {0x3805, 0x20},
    {0x3806, 0x7}, {0x3807, 0x98}, {0x3808, 0xa}, {0x3809, 0x20}, {0x380a, 0x7}, {0x380b, 0x98},
    {0x380c, 0xc}, {0x380d, 0x80}, {0x380e, 0x7}, {0x380f, 0xd0}, {0x5001, 0x7f}, {0x5680, 0x0},
    {0x5681, 0x0}, {0x5682, 0xa}, {0x5683, 0x20}, {0x5684, 0x0}, {0x5685, 0x0}, {0x5686, 0x7},
    {0x5687, 0x98},
    {0xffff, 0xff},
};

void sensor_write_array(const uint16_t data[][2]) {
    int i = 0;
    while (1) {
        uint16_t reg = data[i][0];
        uint8_t  val = (uint8_t)data[i][1];
        if (reg == 0xFFFF && val == 0xFF) break;
        i2c_write_reg16(reg, val);
        i++;
    }
}

// ==========================================
// 6. CAPTURE + SIGNAL-PROBE HELPERS
// ==========================================
// Register read without the 100us settle delay - used only for high-rate
// polling of the TRIG register, where each read is an independent SPI
// transaction anyway. Gets us ~10k samples/sec.
uint8_t arducam_read_reg_fast(uint8_t addr) {
    SPI_REG(0x70) = 0xFFFFFFFE;
    spi_transfer(addr & 0x7F);
    uint8_t val = spi_transfer(0x00);
    SPI_REG(0x70) = 0xFFFFFFFF;
    return val;
}

uint32_t read_fifo_length(void) {
    uint32_t len1 = arducam_read_reg(ARDUCHIP_FIFO_SIZE1);
    uint32_t len2 = arducam_read_reg(ARDUCHIP_FIFO_SIZE2);
    uint32_t len3 = arducam_read_reg(ARDUCHIP_FIFO_SIZE3) & 0x7F;
    return ((len3 << 16) | (len2 << 8) | len1) & 0x07FFFFF;
}

// Samples the LIVE VSYNC level (TRIG bit0) ~20k times back-to-back (~2s).
// At any plausible frame rate (15-60fps) a driven VSYNC line produces
// 30-240 level transitions in that window - impossible to miss. Every
// earlier run only ever looked at the DONE bit; this is the first direct
// measurement of whether a frame-sync signal reaches the shield at all.
uint32_t vsync_probe(void) {
    const uint32_t N = 20000;
    uint32_t highs = 0, transitions = 0;
    uint8_t last = arducam_read_reg_fast(ARDUCHIP_TRIG) & 0x01;
    for (uint32_t i = 0; i < N; i++) {
        uint8_t v = arducam_read_reg_fast(ARDUCHIP_TRIG) & 0x01;
        if (v) highs++;
        if (v != last) { transitions++; last = v; }
    }
    printf("  %u samples: high=%u transitions=%u -> VSYNC %s\n",
           N, highs, transitions,
           transitions ? "IS TOGGLING (sensor is alive!)" : "FLAT (line never moved)");
    return transitions;
}

// One capture attempt with microsecond-scale timing: FIFO counter baseline
// right after CLEAR (is the mysterious "8" already there before capture?),
// then fast-poll DONE and report how many polls it took (each ~0.1ms).
// Instant done (<~10 polls) = stuck-level artifact; ~150-700 polls = one
// real frame time at 15-60fps; timeout = frame never completes.
uint32_t try_capture(void) {
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
    uint32_t base = read_fifo_length();
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_START_MASK);
    const int MAXP = 60000; // ~6s of fast polls
    int i;
    for (i = 0; i < MAXP; i++)
        if (arducam_read_reg_fast(ARDUCHIP_TRIG) & CAP_DONE_MASK) break;
    uint32_t len = read_fifo_length();
    printf("    fifo-after-clear=%u  done=%s (%d fast-polls, ~%dms)  captured=%u bytes\n",
           base, (i < MAXP) ? "yes" : "TIMEOUT", i, i / 10, len);
    return (i < MAXP) ? len : 0;
}

void save_fifo(const char *filename, uint32_t save_len) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) { perror("File open failed"); return; }

    SPI_REG(0x70) = 0xFFFFFFFE; // CS Assert
    spi_transfer(BURST_FIFO_READ);
    spi_transfer(0x00); // dummy byte

    uint8_t seg[4096];
    for (uint32_t i = 0; i < save_len; ) {
        uint32_t m = save_len - i;
        if (m > sizeof(seg)) m = sizeof(seg);
        spi_read_burst(seg, m);
        fwrite(seg, 1, m, fp);
        i += m;
    }

    SPI_REG(0x70) = 0xFFFFFFFF; // CS Deassert
    fclose(fp);
    printf("  Saved %u bytes to '%s'\n", save_len, filename);
}

// Scan for the JPEG frame inside the FIFO readout.
// Returns 2 = SOI+EOI found, 1 = SOI only (truncated), 0 = no JPEG.
int find_jpeg(const uint8_t *buf, uint32_t len, uint32_t *soi, uint32_t *eoi) {
    int have_soi = 0;
    for (uint32_t i = 0; i + 1 < len; i++) {
        // SOI must be followed by another marker (FF Ex/DB/...) - a bare
        // FF D8 also appears in wrongly-aligned streams, a FF D8 FF cannot.
        if (!have_soi && i + 2 < len &&
            buf[i] == 0xFF && buf[i + 1] == 0xD8 && buf[i + 2] == 0xFF) { *soi = i; have_soi = 1; }
        else if (have_soi && buf[i] == 0xFF && buf[i + 1] == 0xD9) { *eoi = i + 2; return 2; }
    }
    return have_soi;
}

// Hard power cycle of the sensor via the shield's LDO enable (ARDUCHIP_GPIO
// bit2), PWDN held high while power comes up per the datasheet. Recovers
// states a 0x3008 soft reset cannot - including the case where the reset
// never happened because I2C writes were dying. If this shield revision
// doesn't wire the LDO enable, it degrades to a harmless PWDN toggle.
void sensor_power_cycle(void) {
    arducam_write_reg(ARDUCHIP_GPIO, GPIO_RESET_MASK | GPIO_PWDN_MASK);  // LDO off
    usleep(300000);
    arducam_write_reg(ARDUCHIP_GPIO, GPIO_RESET_MASK | GPIO_PWDN_MASK | GPIO_PWREN_MASK);
    usleep(100000);
    arducam_write_reg(ARDUCHIP_GPIO, GPIO_RESET_MASK | GPIO_PWREN_MASK); // PWDN low
    usleep(100000);
}

// Write canary: flip 0x4407 (JPEG compression scale - harmless while no
// capture is in progress), read it back, restore. Distinguishes "I2C writes
// silently dying" from every other failure mode: reads can keep working
// perfectly while writes never land, and then even a soft reset fixes
// nothing because the reset command is itself a write. That exact condition
// once made every capture fail with correct-looking fingerprint readbacks.
int sensor_writes_alive(void) {
    uint8_t q0 = i2c_read_reg16(0x4407);
    i2c_write_reg16(0x4407, q0 ^ 0x01);
    uint8_t q1 = i2c_read_reg16(0x4407);
    i2c_write_reg16(0x4407, q0);
    return q1 == (q0 ^ 0x01);
}

// ==========================================
// 6b. JPEG CAPTURE MODE (full sensor resolution)
// ==========================================
// The sensor's on-chip JPEG encoder compresses the frame, so even
// 2592x1944 (~0.3-1.5MB compressed) fits the shield's 8MB FIFO -- raw
// YUV at that size (10.1MB) would not. Output needs no makeimg step:
// the FIFO contents ARE a .jpg file (plus some pad bytes we trim off
// by scanning for the JPEG start/end markers FFD8/FFD9).
int jpeg_capture(const uint16_t (*size_arr)[2], const char *size_desc, uint8_t quality, uint8_t sat, int warm) {
    double t_cfg = now_ms();
    if (!warm) {
    printf("[STEP 3J] Configuring OV5642 for JPEG %s (quality scale 0x%02X, lower=finer)...\n",
           size_desc, quality);
    // DO NOT trim these delays. A build that cut the post-reset settle to
    // 50ms left the sensor half-configured (steady captures of garbage at
    // half the normal frame time) while the late registers still read back
    // correctly - so the warm-start fingerprint matched a broken state.
    // These exact timings are the ones proven good across many captures.
    i2c_write_reg16(0x3008, 0x80);  // software reset
    usleep(200000);
    sensor_write_array(OV5642_QVGA_Preview);
    usleep(100000);
    sensor_write_array(OV5642_JPEG_Capture_QSXGA);
    sensor_write_array(ov5642_320x240);
    usleep(100000);
    // Fixups ArduCAM's InitCAM() applies after the arrays for this module
    i2c_write_reg16(0x3818, 0xa8);  // timing ctl: compression enable + mirror
    i2c_write_reg16(0x3621, 0x10);  // array ctl matching 0x3818 mirror setting
    i2c_write_reg16(0x3801, 0xb0);  // HS low byte matching mirror
    i2c_write_reg16(0x4407, quality);
    i2c_write_reg16(0x5888, 0x00);  // LENC off (ArduCAM default for JPEG)
    i2c_write_reg16(0x5000, 0xff);  // ISP: LENC/GAMMA/BPC/WPC/CIP all on
    sensor_write_array(size_arr);   // final output resolution
    usleep(100000);

    // The size arrays end with 0x5001=0x7f, which clears bit7 - the SDE
    // (special digital effects) enable. SDE is the ISP block that applies
    // the saturation programmed in 0x5583/0x5584, so with it off every
    // JPEG came out muted regardless of those settings. Re-enable it and
    // set saturation LAST so nothing overrides it - same order ArduCAM's
    // own set_Color_Saturation() uses (it too writes 0x5001=0xff first).
    i2c_write_reg16(0x5001, 0xff);
    i2c_write_reg16(0x5583, sat);   // U (blue-diff) saturation gain
    i2c_write_reg16(0x5584, sat);   // V (red-diff)  saturation gain
    i2c_write_reg16(0x5580, 0x02);  // SDE ctrl: saturation enable

    // Readback: catches both a wrong config and the silent bus-death mode
    // seen once before (everything reading back as one stale byte).
    uint8_t v3818 = i2c_read_reg16(0x3818);
    uint8_t v4407 = i2c_read_reg16(0x4407);
    uint8_t v3808 = i2c_read_reg16(0x3808);
    uint8_t v3809 = i2c_read_reg16(0x3809);
    printf("  readback 0x3818=0x%02X (want 0xA8)  0x4407=0x%02X (want 0x%02X)  width=0x%02X%02X px\n",
           v3818, v4407, quality, v3808, v3809);
    if (v3818 != 0xa8 || v4407 != quality) {
        fprintf(stderr, "[WARN] config readback mismatch: 0x3818=0x%02X (want A8) 0x4407=0x%02X (want %02X)\n",
                v3818, v4407, quality);
    }
    } // !warm

    printf("[STEP 4J] Shield init...\n");
    arducam_write_reg(ARDUCHIP_TIM, VSYNC_LEVEL_MASK); // pairing proven by the YUV path
    arducam_write_reg(ARDUCHIP_FRAMES, 0x00);          // 1 frame
    // Reference flow clears twice (flush_fifo + clear_fifo_flag); we also
    // force both FIFO pointers to zero - stale byte counts have been seen
    // carrying over between runs, which would make the burst read return
    // leftover bytes from the PREVIOUS capture instead of the new JPEG.
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
    double t_settle = now_ms();
    // Cold start: AE/AWB begin from reset defaults and need ~1.5s of live
    // frames to converge. Warm start: the sensor kept free-running since the
    // previous capture so AE is already converged on the scene; 300ms grace
    // covers a camera that was just repositioned.
    usleep(warm ? 300000 : 1500000);

    printf("[STEP 5J] Capturing...\n");
    double t_cap = now_ms();
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
    uint32_t len = try_capture();
    if (len <= 1024 || len >= 0x7FFFF0) {
        // Same polarity fallback the YUV matrix used
        printf("  retrying with shield TIM=0x00...\n");
        arducam_write_reg(ARDUCHIP_TIM, 0x00);
        usleep(100000);
        len = try_capture();
    }
    if (len <= 1024 || len >= 0x7FFFF0) {
        fprintf(stderr, "[FAIL] no plausible frame in FIFO (len=%u)%s\n", len,
                len >= 0x7FFFF0 ? " - SPI read all-0xFF, power-cycle and reseat the module" : "");
        // Leave the shield in a clean state: the fallback above may have left
        // TIM=0x00 and a still-pending capture, which corrupted the runs that
        // followed a failed one before this cleanup existed.
        arducam_write_reg(ARDUCHIP_TIM, VSYNC_LEVEL_MASK);
        arducam_write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK | FIFO_WRPTR_RST_MASK);
        arducam_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
        return -1;
    }

    printf("[STEP 6J] Reading %u bytes from FIFO...\n", len);
    // Read the stream INCLUDING the first post-command byte (previously
    // discarded as a dummy). Byte-level analysis of real captures showed
    // this shield's 8MB FIFO sometimes returns 16-bit words BYTE-SWAPPED
    // and offset by one (real[0] carried in the "dummy" slot), and
    // sometimes as-is - it varies with the FIFO pointer phase, so the
    // alignment is auto-detected below rather than assumed. (The same swap
    // makes the YUV path's YUYV stream read as UYVY - makeimg5.py
    // compensates there.) Two spare bytes complete the final word.
    uint8_t *s   = malloc(len + 2);
    uint8_t *buf = malloc(len + 2);
    if (!s || !buf) { printf("[FAIL] malloc(%u) failed\n", len + 2); free(s); free(buf); return -1; }
    double t_read = now_ms();
    SPI_REG(0x70) = 0xFFFFFFFE;
    spi_transfer(BURST_FIFO_READ);
    for (uint32_t off = 0; off < len + 2; ) {
        uint32_t seg = (len + 2) - off;
        if (seg > 0x40000) seg = 0x40000;
        spi_read_burst(s + off, seg);
        off += seg;
        printf("  ... %u KB\n", off / 1024);
        fflush(stdout);
    }
    SPI_REG(0x70) = 0xFFFFFFFF;
    double t_done = now_ms();
    fprintf(stderr, "[TIME] config=%.2fs settle=%.2fs capture=%.2fs readout=%.2fs (%u bytes, %.1fus/byte) [%s]\n",
            (t_settle - t_cfg) / 1000.0, (t_cap - t_settle) / 1000.0,
            (t_read - t_cap) / 1000.0, (t_done - t_read) / 1000.0,
            len + 2, (t_done - t_read) * 1000.0 / (len + 2),
            warm ? "warm" : "cold");

    printf("  first 16 raw bytes:");
    for (uint32_t d = 0; d < 16 && d < len; d++) printf(" %02X", s[d]);
    printf("\n");

    // Reconstruct the true byte order. The FIFO's word alignment varies
    // run-to-run with the pointer phase: sometimes the stream arrives
    // as-is, sometimes 16-bit byte-swapped (both observed on hardware).
    // The strict FF D8 FF test in find_jpeg picks the right one - a
    // wrongly-aligned stream cannot produce that sequence at the front.
    // Pass 0 requires the frame to start near the front; pass 1 relaxes
    // that for streams with stale FIFO bytes ahead of the frame, but then
    // insists on a COMPLETE frame (SOI and EOI) to stay trustworthy.
    static const char *how[3] = { "direct (as-read)", "word un-swap, odd phase", "word un-swap, even phase" };
    uint32_t soi = 0, eoi = 0, blen = 0;
    int found = 0, cand = -1;
    for (int pass = 0; pass < 2 && !found; pass++) {
        for (int c = 0; c < 3 && !found; c++) {
            if (c == 0) {
                memcpy(buf, s, len + 2);   // stream exactly as read
                blen = len + 2;
            } else if (c == 1) {
                buf[0] = s[0];
                for (uint32_t i = 1; i + 1 < len + 2; i += 2) { buf[i] = s[i + 1]; buf[i + 1] = s[i]; }
                blen = len + 1;
                // first byte can be clipped from its half-word; it is always FF
                if (buf[0] != 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF) buf[0] = 0xFF;
            } else {
                for (uint32_t i = 0; i + 1 < len + 2; i += 2) { buf[i] = s[i + 1]; buf[i + 1] = s[i]; }
                blen = len + 2;
            }
            found = find_jpeg(buf, blen, &soi, &eoi);
            if (found && pass == 0 && soi > 64) found = 0; // frame must begin near the front
            if (found && pass == 1 && found != 2) found = 0; // deep SOI: complete frames only
            if (found) cand = c;
        }
    }

    if (!found) {
        fprintf(stderr, "[FAIL] no JPEG marker in any alignment (%u bytes) - raw stream in fifo_dump.bin\n",
                len + 2);
        fprintf(stderr, "       first bytes:");
        for (uint32_t d = 0; d < 16 && d < len; d++) fprintf(stderr, " %02X", s[d]);
        fprintf(stderr, "  capture=%s frame\n", (t_read - t_cap) < 150.0 ? "SHORT (sensor misconfigured)" : "normal-length");
        FILE *fp = fopen("fifo_dump.bin", "wb");
        if (fp) { fwrite(s, 1, len + 2, fp); fclose(fp); }
        free(s); free(buf);
        return -1;
    }
    free(s);
    printf("  readout alignment: %s (start marker at offset %u)\n", how[cand], soi);
    if (found == 1) {
        printf("  [WARN] start marker found but no end marker - frame is truncated;\n");
        printf("  saving anyway (viewers decode truncated JPEGs fine).\n");
        eoi = blen;
    }
    FILE *fp = fopen("image.jpg", "wb");
    if (!fp) { perror("File open failed"); free(buf); return -1; }
    fwrite(buf + soi, 1, eoi - soi, fp);
    fclose(fp);
    printf("\n[SUCCESS] Saved %u-byte JPEG (%s) to 'image.jpg'\n", eoi - soi, size_desc);
    printf("scp it over and open directly - no makeimg step needed.\n");
    free(buf);
    fprintf(stderr, "[TIME] post=%.2fs total=%.2fs -> image.jpg (%u bytes)\n",
            (now_ms() - t_done) / 1000.0, (now_ms() - g_t0) / 1000.0, eoi - soi);
    return 0;
}

// ==========================================
// 7. MAIN EXECUTION
// ==========================================
int main(int argc, char **argv) {
    g_t0 = now_ms();
    // Mode select: no args = original QVGA YUV path (unchanged, feeds
    // makeimg5.py); "jpeg [size] [quality]" = on-sensor JPEG at up to 5MP.
    int jpeg_mode = 0, force_fresh = 0;
    const uint16_t (*size_arr)[2] = ov5642_2592x1944;
    const char *size_desc = "2592x1944 (full 5MP)";
    uint8_t quality = 0x08;
    uint8_t sat = 0x60;   // SDE saturation gain; sat0..sat4 -> 0x40..0x80 (ArduCAM's own scale)
    if (argc >= 2) {
        if (strcmp(argv[1], "jpeg") != 0) {
            printf("Usage: %s              -> 320x240 raw YUV to image.bin (decode with makeimg5.py)\n", argv[0]);
            printf("       %s jpeg [size] [q]\n", argv[0]);
            printf("  size: 320 640 1024 1280 1600 2048 2592   (default 2592 = full 5MP)\n");
            printf("  q:    2-63 JPEG quantization scale, lower = finer (default 8)\n");
            return -1;
        }
        jpeg_mode = 1;
        for (int a = 2; a < argc; a++) {
            if      (!strcmp(argv[a], "fresh") || !strcmp(argv[a], "cold")) force_fresh = 1;
            else if (!strncmp(argv[a], "sat", 3) && argv[a][3] >= '0' && argv[a][3] <= '4' && !argv[a][4])
                sat = (uint8_t)(0x40 + (argv[a][3] - '0') * 0x10);
            else if (!strcmp(argv[a], "320"))  { size_arr = ov5642_320x240;   size_desc = "320x240"; }
            else if (!strcmp(argv[a], "640"))  { size_arr = ov5642_640x480;   size_desc = "640x480"; }
            else if (!strcmp(argv[a], "1024")) { size_arr = ov5642_1024x768;  size_desc = "1024x768"; }
            else if (!strcmp(argv[a], "1280")) { size_arr = ov5642_1280x960;  size_desc = "1280x960"; }
            else if (!strcmp(argv[a], "1600")) { size_arr = ov5642_1600x1200; size_desc = "1600x1200"; }
            else if (!strcmp(argv[a], "2048")) { size_arr = ov5642_2048x1536; size_desc = "2048x1536"; }
            else if (!strcmp(argv[a], "2592")) { /* default */ }
            else {
                int q = atoi(argv[a]);
                if (q >= 2 && q <= 63) quality = (uint8_t)q;
                else { printf("Unknown arg '%s' (sizes 320..2592, q 2-63, sat0-sat4, or 'fresh')\n", argv[a]); return -1; }
            }
        }
    }

    printf("[INIT] Mapping Physical Memory...\n");
    spi_base_virt = (volatile uint32_t *)map_physical_memory(SPI_PHYS_BASE);
    i2c_base_virt = (volatile uint32_t *)map_physical_memory(I2C_PHYS_BASE);

    spi_init();
    i2c_init();

    // 1. Check SPI
    printf("[STEP 1] Check SPI... ");
    arducam_write_reg(ARDUCHIP_TEST1, 0x55);
    uint8_t test = arducam_read_reg(ARDUCHIP_TEST1);
    if(test != 0x55) { fprintf(stderr, "[FAIL] shield SPI test: wrote 0x55, read 0x%02X\n", test); return -1; }
    printf("PASS\n");
    uint8_t rev = arducam_read_reg(ARDUCHIP_REV);
    printf("  ARDUCHIP_REV: 0x%02X (ver_low=0x%02X ver_high=0x%02X)\n",
           rev, rev & VER_LOW_MASK, (rev & VER_HIGH_MASK) >> 6);

    // Warm-start fast path (jpeg mode only). The sensor free-runs between
    // invocations and keeps every register plus its converged AE/AWB state,
    // so if it still carries exactly the requested JPEG config from a
    // previous run there is nothing to configure: skip the PWDN sequencing
    // (~0.16s), the ~650-register re-init (~0.9s) and most of the AE settle
    // (~1.2s). First run after power-up, a changed size/quality, or a hung
    // sensor all fail this check and fall through to the cold path below.
    if (jpeg_mode && force_fresh) {
        // User-requested clean slate: hard LDO power cycle now, then the
        // normal PWDN sequencing + ID check + full cold init run below and
        // validate the sensor came back. Clears any latched calibration
        // state (black level, AWB history) a warm run would carry over.
        fprintf(stderr, "[FRESH] hard sensor power cycle + full re-init requested\n");
        sensor_power_cycle();
    }
    if (jpeg_mode && !force_fresh) {
        uint8_t wid_h = 0, wid_l = 0;
        for (int i = 0; size_arr[i][0] != 0xFFFF; i++) {
            if (size_arr[i][0] == 0x3808) wid_h = (uint8_t)size_arr[i][1];
            if (size_arr[i][0] == 0x3809) wid_l = (uint8_t)size_arr[i][1];
        }
        if (i2c_read_reg16(OV5642_CHIPID_HIGH) == 0x56 &&
            i2c_read_reg16(OV5642_CHIPID_LOW)  == 0x42 &&
            i2c_read_reg16(0x3818) == 0xa8 &&     // JPEG-mode fixups present
            i2c_read_reg16(0x4407) == quality &&  // same compression scale
            i2c_read_reg16(0x3808) == wid_h &&    // same output width
            i2c_read_reg16(0x3809) == wid_l &&
            i2c_read_reg16(0x5001) == 0xff &&     // SDE (color) still enabled
            i2c_read_reg16(0x5583) == sat) {      // same saturation setting
            printf("[WARM] Sensor already configured for JPEG %s - skipping re-init.\n", size_desc);
            if (jpeg_capture(size_arr, size_desc, quality, sat, 1) == 0)
                return 0;
            // The fingerprint can match a sensor whose earlier (timing/PLL)
            // registers are broken - e.g. after an interrupted or failed
            // cold init. Never trust warm state after a failure: fall
            // through to the full PWDN + reset + re-init cold path.
            fprintf(stderr, "[WARM] warm capture failed - falling back to full cold re-init\n");
        }
    }

    // 1.5 OV5642 datasheet section 2.6.1 ("power up with internal DVDD and
    // I2C access during power up period") requires: PWDN must be held HIGH
    // for as long as I2C may be accessed before the sensor's own power rails
    // are stable, and only brought low afterward (AVDD-to-PWDN >= 1ms). Our
    // previous build skipped straight to "PWDN low" - if the sensor's very
    // first power-up ever violated this ordering, it can latch into an
    // undefined state that a register-level soft reset (0x3008=0x80) cannot
    // clear, since that reset itself depends on I2C already working. This
    // won't undo a bad latch from the past (only a real power cycle can),
    // but it's the datasheet-correct sequence going forward.
    printf("[STEP 1.5] Sequencing sensor PWDN per datasheet (hold high, then release)...\n");
    arducam_write_reg(ARDUCHIP_GPIO, GPIO_RESET_MASK | GPIO_PWDN_MASK | GPIO_PWREN_MASK); // PWDN=HIGH (standby)
    usleep(100000); // generous margin over the datasheet's 1ms minimum
    arducam_write_reg(ARDUCHIP_GPIO, GPIO_RESET_MASK | GPIO_PWREN_MASK);                  // PWDN=LOW (normal)
    usleep(50000);
    printf("  ARDUCHIP_GPIO readback: 0x%02X\n", arducam_read_reg(ARDUCHIP_GPIO));

    // 2. Confirm the OV5642 actually answers on I2C before wasting time
    // writing ~530 registers into the void. Chip ID should read 0x56 / 0x42.
    printf("[STEP 2] Checking OV5642 I2C ID... ");
    uint8_t id_h = i2c_read_reg16(OV5642_CHIPID_HIGH);
    uint8_t id_l = i2c_read_reg16(OV5642_CHIPID_LOW);
    printf("(0x%02X%02X) ", id_h, id_l);
    if (id_h != 0x56 || id_l != 0x42) {
        fprintf(stderr, "[FAIL] OV5642 chip ID read 0x%02X%02X (want 0x5642) - sensor not responding on I2C\n",
                id_h, id_l);
        return -1;
    }
    printf("PASS\n");

    if (jpeg_mode) {
        // Cold path with escalating recovery. First make sure writes land at
        // all (they can die while reads stay perfect); then a failed capture
        // gets ONE hard sensor power cycle + full re-init before giving up.
        if (!sensor_writes_alive()) {
            fprintf(stderr, "[FAIL] sensor ACKs reads but writes do not land - power-cycling sensor via shield LDO\n");
            sensor_power_cycle();
            if (i2c_read_reg16(OV5642_CHIPID_HIGH) != 0x56 || !sensor_writes_alive()) {
                fprintf(stderr, "[FAIL] writes still dead after LDO power cycle - physically power-cycle the board\n");
                return -1;
            }
        }
        if (jpeg_capture(size_arr, size_desc, quality, sat, 0) == 0)
            return 0;
        fprintf(stderr, "[RETRY] cold capture failed - sensor power cycle + one more full init\n");
        sensor_power_cycle();
        if (i2c_read_reg16(OV5642_CHIPID_HIGH) != 0x56) {
            fprintf(stderr, "[FAIL] sensor did not come back after power cycle - check the module\n");
            return -1;
        }
        return jpeg_capture(size_arr, size_desc, quality, sat, 0);
    }

    // 3. Configure Sensor
    printf("[STEP 3] Configuring OV5642... ");
    i2c_write_reg16(0x3008, 0x80);  // Software reset (self-clearing)
    usleep(200000);
    sensor_write_array(OV5642_QVGA_Preview);
    usleep(100000);

    // Keep the array's native output config: it already sets YUV422
    // (0x501f=0x00 format mux, 0x4300=0x30 YUYV order) at 320x240 - the
    // same pixel layout makeimg.py decodes. An earlier build layered
    // ArduCAM's BMP-branch overrides on top (0x501f=0x01/0x4300=0x61); I
    // had mislabeled those as YUV422 - they are actually RGB565. Reverted
    // so the stream stays YUV. The two read-modify-writes below
    // (mirror/flip) mirror ArduCAM's InitCAM() tail for non-JPEG modes.
    uint8_t reg_val = i2c_read_reg16(0x3818);
    printf("\n  readback 0x3818=0x%02X (array wrote 0xC1)", reg_val);
    i2c_write_reg16(0x3818, (reg_val | 0x60) & 0xFF);
    reg_val = i2c_read_reg16(0x3621);
    printf("  readback 0x3621=0x%02X (array wrote 0x87)\n", reg_val);
    i2c_write_reg16(0x3621, reg_val & 0xDF);
    printf("  DONE.\n");

    // 3.6 THE DECISIVE CHECK. Reads are proven good (chip ID), but sensor
    // WRITES have been fire-and-forget this whole time - never once verified.
    // The OV5642 powers up with its DVP output pads DISABLED (0x3017/0x3018
    // default off), so if writes are silently failing the sensor never drives
    // VSYNC/HREF/PCLK at all -> the shield sees a dead bus -> instant tiny
    // "done" captures, immune to every config change. That matches every
    // symptom so far exactly. Read back critical registers, retrying the
    // write up to 3x on mismatch.
    static const struct { uint16_t reg; uint8_t want; const char *desc; } verify[] = {
        {0x3017, 0x7f, "pad enable 1 (VSYNC/HREF/PCLK drivers)"},
        {0x3018, 0xfc, "pad enable 2 (pixel data pins)"},
        {0x3808, 0x01, "output width  high byte (0x0140=320)"},
        {0x3809, 0x40, "output width  low byte"},
        {0x501f, 0x00, "format mux (YUV passthrough)"},
        {0x4300, 0x30, "output format YUV422 (YUYV)"},
        {0x3008, 0x02, "system control (normal, not sleeping)"},
        {0x300e, 0x18, "interface select (DVP out, MIPI off)"},
    };
    int mismatches = 0;
    printf("[STEP 3.6] Verifying sensor writes actually landed...\n");
    for (unsigned v = 0; v < sizeof(verify)/sizeof(verify[0]); v++) {
        uint8_t got = i2c_read_reg16(verify[v].reg);
        int tries = 0;
        while (got != verify[v].want && tries < 3) {
            i2c_write_reg16(verify[v].reg, verify[v].want);
            got = i2c_read_reg16(verify[v].reg);
            tries++;
        }
        int ok = (got == verify[v].want);
        if (!ok) mismatches++;
        printf("  reg 0x%04X = 0x%02X (want 0x%02X) %-9s %s%s\n",
               verify[v].reg, got, verify[v].want,
               ok ? "OK" : "MISMATCH", verify[v].desc,
               tries ? " [retried]" : "");
    }

    // 4. Shield-side init.
    printf("[STEP 4] Shield init...\n");
    arducam_write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
    arducam_write_reg(ARDUCHIP_FRAMES, 0x00); // capture 1 frame
    usleep(500000);                           // let sensor AEC/AWB settle

    // 5. THE key measurement no run has made yet: is the sensor's frame-sync
    // line electrically moving at all? (Independent of any polarity setting -
    // a toggling line shows transitions under either polarity.)
    printf("[STEP 5] Live VSYNC activity probe (~2s of fast sampling)...\n");
    uint8_t v4740 = i2c_read_reg16(0x4740);
    printf("  sensor 0x4740 (DVP timing/polarity ctl) = 0x%02X\n", v4740);
    uint32_t vsync_trans = vsync_probe();

    // 6. Polarity matrix: sensor-side VSYNC/PCLK polarity (0x4740) x
    // shield-side VSYNC interpretation (ARDUCHIP_TIM). Earlier runs only
    // ever tried two of these four combinations (array-default+TIM=0 and
    // 0x21+TIM=2); if the pairing is mismatched the shield sees "frame
    // over" instantly - exactly the 8-byte signature.
    printf("[STEP 6] Capture attempts across full polarity matrix...\n");
    const uint8_t tims[2] = {0x00, VSYNC_LEVEL_MASK};
    uint8_t v47s[2];
    v47s[0] = v4740;  // whatever the init array left (sensor default)
    v47s[1] = 0x21;   // value ArduCAM's own driver uses for DVP capture
    uint32_t final_len = 0;
    for (int s = 0; s < 2 && !final_len; s++) {
        for (int t = 0; t < 2 && !final_len; t++) {
            i2c_write_reg16(0x4740, v47s[s]);
            arducam_write_reg(ARDUCHIP_TIM, tims[t]);
            usleep(100000);
            printf("  [sensor 0x4740=0x%02X, shield TIM=0x%02X]\n", v47s[s], tims[t]);
            uint32_t len = try_capture();
            if (len > 4096 && len < 0x07FFFFF) final_len = len; // plausible frame
        }
    }

    if (final_len) {
        printf("\n[SUCCESS] Plausible frame captured: %u bytes! Saving...\n", final_len);
        save_fifo("image.bin", final_len > 400000 ? 400000 : final_len);
        printf("Decode with makeimg5.py (UYVY byte order + stale-header skip).\n");
        printf("For full-quality captures run: %s jpeg\n", "./camcap5");
        return 0;
    }

    printf("\n===== SUMMARY =====\n");
    if (mismatches > 0) {
        printf("VERDICT: sensor register writes did not stick (STEP 3.6) - software issue,\n");
        printf("report this full output.\n");
    } else if (vsync_trans == 0) {
        printf("VERDICT: sensor control interface fully works and is verified configured,\n");
        printf("but its frame-sync (VSYNC) line NEVER moved during ~2s of monitoring, in\n");
        printf("every polarity configuration. The sensor's pixel-output side is silent\n");
        printf("while its control side is alive. On this module those output lines run\n");
        printf("from the sensor head through the small flex ribbon/connector by the lens.\n");
        printf("ACTION: power off, gently reseat that flex (open latch, remove, wipe the\n");
        printf("gold contacts, reinsert firmly), then rerun. If VSYNC still reads FLAT\n");
        printf("after reseating, the module hardware is faulty.\n");
    } else {
        printf("VERDICT: VSYNC IS toggling (%u transitions) - the sensor is producing\n", vsync_trans);
        printf("frames! But no pixel bytes reach the FIFO in any polarity combination,\n");
        printf("so PCLK/HREF/data lines are not getting through (flex-cable contact or\n");
        printf("ArduChip issue). Reseat the sensor flex and rerun. Report this output -\n");
        printf("a live VSYNC significantly changes the diagnosis.\n");
    }

    return 0;
}
