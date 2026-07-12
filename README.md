# C2S

https://digilent.com/reference/programmable-logic/arty-a7/reference-manual?redirect=1


Full procedure : 

    1) Connect jtag + ethernet and flash the bitstream 
    
    2) You will require 4 serial terminal tabs (for convinience sake)
    
    3) In tab 1 : ensure your pwd is VEGA/vega-tools/utils/eth_transfer/accelerator where you will have bbl + dts : "./send.sh bbl.bin riscv.dtb"
    
    4) In tab 2 : with any pwd, do "minicom trishul32" to open the uart to the board
    
    5) In tab 3 : Perform all C code compilation and network configurations from this tab (more about that later)
    
    6) Once the tab 2 shows the full bootloader sequence, your username + pwd is root
    
    7) In tab 4 : Perform all image actions here : all jpeg to binary and vice versa conversions here (more about that later)

    8) In tab 2 : Once into the linux : we can configure the networking (on board side) with these commands in order : 
      a) rm *
      b) ip link set eth0 up
 	    c) ip addr add 192.168.1.20/24 dev eth0

    9) In tab 3 : we can configure the networking (on host side) with these commands in order : 
      a) sudo ip addr flush dev enp1s0
      b) sudo ip addr add 192.168.1.10/24 dev enp1s0
      c) sudo ip link set enp1s0 up
      d) ssh-keygen -f '/home/amrut/.ssh/known_hosts' -R '192.168.1.20'
      
    10) Verify by "ping 192.168.1.10" from board side in tab 2

    11) In tab 3 : enter into the directory in which your c code sits, (let's call it whatever.c), perform the compilation using the built compiler toolchain
    
    a) VEGA/trisul32_buildroot/buildroot/output/host/bin/riscv32-linux-gcc <whatever>.c -o <whatever> -static -lm
    b) VEGA/trisul32_buildroot/buildroot/output/host/bin/riscv32-linux-strip <whatever> 
    c) scp -O <whatever> root@192.168.1.20:/root/

    12) In tab 2 : verify whether the compiled binary has been uploaded or not : "ls *"

    13) In tab 2 : perform these commands : 
    
        a) chmod +x whatever
        b) ./whatever

    14) In tab 3 : if you want to copy files from board to host : scp -O root@192.168.1.20:/root/filename /home/...where you want it

    

