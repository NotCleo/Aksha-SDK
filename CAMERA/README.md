<img width="1783" height="517" alt="image" src="https://github.com/user-attachments/assets/b809e96e-ee4a-44dd-aeaa-53d103547f14" />

Your best general-purpose command (mid/high altitude):
./PM image.jpg weights.bin --conf 0.50
Low-altitude / close subject:
./PM image.jpg weights.bin --conf 0.42 --stride 90 --scales 0.28,0.5

<img width="1783" height="405" alt="image" src="https://github.com/user-attachments/assets/b813e966-b420-424e-8ee1-748759e408cf" />


./camera jpeg 1024 8 (or ./camera jpeg 1280 8 at low altitude). Drop 12→8. sat doesn't matter for detection.

<img width="1770" height="612" alt="image" src="https://github.com/user-attachments/assets/723fed5a-5ac1-4e96-9fa6-ff0b94db6571" />
