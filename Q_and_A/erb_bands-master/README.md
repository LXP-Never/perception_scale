# erb_bands
https://github.com/flavioeverardo/erb_bands/blob/master/erb.py

ERB representation of an audio file implemented in Python.

## Description
An audio file decomposed in samples after STFT and it is represented in (user-defined) ERB bands.
This implementatio does not use a complete auditory model.

The main ideas, code and concepts to read the wav files and perform the STFT is taken from Ajin Tom in his repo:
https://github.com/ajintom/auto-spatial </br>
On the other hand, the main ideas, code and concepts to build the ERB bands is taken from William J. Wilkinson in his repo:
https://github.com/wil-j-wil/py_bank

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Requirements
Tested using Travis CI under Linux and Mac OS with Python 2.7 and 3.6.

## Usage
```bash
$ python main.py --erb=40 --file=snare.wav --samples=32768
```

## Examples

kick.wav </br>
Samples: 32768 </br>
ERB bands: 50 </br>
![Screenshot](figure/kick.png)

snare.wav </br>
Samples: 32768 </br>
ERB bands: 20 </br>
![Screenshot](figure/snare.png)

hihats.wav </br>
Samples: 4000 </br>
ERB bands: 40 </br>
![Screenshot](figure/hihats.png)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
