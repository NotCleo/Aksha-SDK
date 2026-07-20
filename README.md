# Digital India RISC-V (DIR-V) Grand Challenge - Chip to Startup Initiative

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
    │   ├── main.c                          # Main Orchestrator 
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

