# Standard SD module wiring (Arduino Uno/Nano, ATmega328P)

| SD module | Board |
|---|---|
| CS | D10 |
| MOSI | D11 |
| MISO | D12 |
| SCK | D13 |
| GND | GND |
| VCC | Follow the SD module specification |

The generated firmware uses `noodle_fs_init(10)`. SPI pins are the Uno/Nano fixed hardware SPI pins; no manual SPI pin configuration is needed. Copy all generated `sdcard/*.bin` files to the FAT32 card root. For bare microSD sockets use 3.3 V supply and level shifting; a breakout module designed for 5 V logic can be powered as its documentation specifies.
