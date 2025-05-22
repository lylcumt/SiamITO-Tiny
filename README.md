# SiamlTo-Tiny

## 📄 **Edge real-time tracking and FPGA-based hardware implementation for infrared tiny object**
> *(Paper Title)*

## 🧠 Network Architecture
![Network Architecture](architecture.jpg)

## 🛠️ Training Environment  
| Component             | Specification               |
|------------------|-------------------------------|
| **GPU**          | NVIDIA RTX 3080 (10GB VRAM)   |
| **CPU**          | Intel i9-11900K @ 3.5GHz      |
| **OS**      | Ubuntu 18.04 LTS              |
| **Python**       | 3.7 (Anaconda Environment)   |
| **PyTorch/CUDA** | 1.8.0 + CUDA 11.1             |

## 📁 Project Structure
```bash
├── SiamITO-Tiny/           # Core tracking code (SiamlTo-Tiny implementation)
├── score=0.9206/          # Experimental results and evaluation metrics
├── assets/         # (Optional) figures such as network diagrams
├── README.md       # Project description
