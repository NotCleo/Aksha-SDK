<img width="1783" height="517" alt="image" src="https://github.com/user-attachments/assets/b809e96e-ee4a-44dd-aeaa-53d103547f14" />

Your best general-purpose command (mid/high altitude):
./PM image.jpg weights.bin --conf 0.50
Low-altitude / close subject:
./PM image.jpg weights.bin --conf 0.42 --stride 90 --scales 0.28,0.5

<img width="1783" height="405" alt="image" src="https://github.com/user-attachments/assets/b813e966-b420-424e-8ee1-748759e408cf" />

320×240 · 640×480 · 1024×768 · 1280×960 · 1600×1200 · 2048×1536 · 2592×1944

./camera jpeg 1024 8 (or ./camera jpeg 1280 8 at low altitude). Drop 12→8. sat doesn't matter for detection.

<img width="1770" height="612" alt="image" src="https://github.com/user-attachments/assets/723fed5a-5ac1-4e96-9fa6-ff0b94db6571" />


    ├── README.md                      
    ├── LICENSE                      
    ├── Documentation                   # Operating Guide
    │   ├── Interfacing-schematics.pdf       # Sensor Interfacing Schematics
    │   ├── board-schematics.pdf             # FPGA Schematics
    │   └── user-guide.pdf                   # Complete User Guide
    ├── Linux-Image                     # Linux 
    │   ├── send.sh                         # Booting Orchestrator
    │   ├── bbl.bin                         # Linux Image (RISC-V Proxy Kernel and Boot Loader)
    │   └── riscv.dts                       # Linux Device Tree 
    ├── Aksha-SDK                       # SDK Orchestrator
    │   ├── main.c                            # Main Orchestrator 
    │   ├── main                            # Main Orchestrator Binary
    │   ├── camera.c                        # Image Capture Orchestrator          
    │   ├── camera                          # Image Capture Orchestrator Binary
    │   ├── stb_image.h                     # C library to decode and load images into memory       
    │   ├── person-model.c                  # Person Model Orchestrator  
    │   ├── person-model.h                  # Person Model Orchestrator Header Definitions
    │   ├── person-model                    # Person Model Orchestrator Binary
    │   ├── person-model-weights.bin        # Person Model Weights Binary         
    │   ├── track-model.c                   # Track Model Orchestrator  
    │   ├── track-model.h                   # Track Model Orchestrator Header Definitions
    │   ├── track-model                     # Track Model Orchestrator Binary
    │   ├── track-model-weights.bin         # Track Model Weights Binary
    │   ├── SD-Write.c                      # SD Card Writing Orchestrator
    │   ├── SD-Write                        # SD Card Writing Orchestrator Binary
    │   ├── GPS.c                           # GPS Telemetry Orchestrator
    │   ├── GPS                             # GPS Telemetry Orchestrator Binary
    │   ├── bluetooth.c                     # Bluetooth Communication Orchestrator
    |   └── bluetooth                       # Bluetooth Communication Orchestrator Binary
    ├── Vivado-FPGA                     # SDK Orchestrator
    └──   └── aksha.mcs                     # Bitstream Memory Configuration (QSPI)


--scales 0.60 --stride 80 --conf 0.67
