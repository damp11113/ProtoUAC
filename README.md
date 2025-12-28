# UAC-Prototype (concept)

A Ultimate Audio Coding prototype in python. This codec is plan to use compression algorithm **ADPCM / MDCT-Hybrid Subband**. This version is using SBR (Spectral Band Replication) Harmonic Replication and Noise Replication. And Parametric Stereo coding. 

Email to `contact@damp11113.xyz` for get STD_TEST file for testing this codec.

# Technology
- [MPS](https://github.com/damp11113/ProtoUAC/blob/UAC-E/parametric_coding.py) (Multiband Parametric Stereo)
- [HPS](https://github.com/damp11113/ProtoUAC/blob/UAC-E/hps.py) (Hybrid Parametric Stereo)
- [HBB](https://github.com/damp11113/ProtoUAC/blob/UAC-E/hbb.py) (Hybrid Baseband)
- [PHSC](https://github.com/damp11113/ProtoUAC/blob/UAC-E/phsc.py) (Parametric Harmonic Stereo Coding)
- [PHXC](https://github.com/damp11113/ProtoUAC/blob/UAC-E/phxc.py) (Parametric Harmonic eXtraction Coding)

# To Do
Profiles
- UAC-LC (Ultimate Audio Coding - Low Complex) 
- UAC-LR (Ultimate Audio Coding - Low Resource) 
- UAC-E (Ultimate Audio Coding - Efficiency) 🟨 Current
- UAC-SE (Ultimate Audio Coding - Superior Efficiency)
- UAC-L (Ultimate Audio Coding - Lossless)
- UAC-B (Ultimate Audio Coding - Broadcast)
- UAC-V (Ultimate Audio Coding - Voice)
- UAC-VLL (Ultimate Audio Coding - Voice Low Latency)

Feature
- Noise Suppression (UAC-V)
- Dynamic Range Compression (UAC-LC, E, SE, B)
- Psychoacoustic

Bitrate Mode
- CBR ✅
- VBR 🟨
- CVBR
- ABR

# Demo
SBR + MPS: https://www.youtube.com/watch?v=VFwbHifd4kU
