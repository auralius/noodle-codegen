#pragma once

// R15/Uno UART0 driver: polling, no RX/TX ring buffers, no dynamic allocation.
// AVR ATmega328P UART0 at F_CPU=16000000, 115200 baud (U2X0).
// Deliberately tiny surface matching the Serial calls in src/main.cpp.
// The Python host must send IMG\n only after READY and wait for each K ACK.
#if !defined(__AVR_ATmega328P__)
#error "This UART0 experiment targets the ATmega328P (Uno/Nano) only"
#endif

#include <Arduino.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <stdint.h>

class R15BareUart {
public:
  void begin(uint32_t baud) {
    // Doubled UART speed yields UBRR=16 at 16 MHz / 115200 baud.
    // General computation also supports other baud values.
    const uint16_t divisor =
        (uint16_t)((F_CPU + (4UL * baud)) / (8UL * baud) - 1UL);
    UCSR0B = 0;                               // RX/TX disabled while configuring
    UCSR0A = _BV(U2X0);                       // double-speed baud generator
    UBRR0H = (uint8_t)(divisor >> 8);
    UBRR0L = (uint8_t)divisor;
    UCSR0C = _BV(UCSZ01) | _BV(UCSZ00);       // 8 data bits, no parity, 1 stop
    UCSR0B = _BV(RXEN0) | _BV(TXEN0);         // RX/TX enabled, no UART IRQ
  }

  int available() const { return (UCSR0A & _BV(RXC0)) ? 1 : 0; }

  int read() {
    if (!(UCSR0A & _BV(RXC0))) return -1;
    return (int)UDR0;                        // consume single UART HW byte
  }

  size_t write(uint8_t value) {
    while (!(UCSR0A & _BV(UDRE0))) {}
    UCSR0A = _BV(U2X0) | _BV(TXC0);           // clear TX-complete status
    UDR0 = value;
    return 1;
  }

  void flush() {
    while (!(UCSR0A & _BV(TXC0))) {}         // wait until last byte fully sent
  }

  void print(const __FlashStringHelper* str) {
    const char* p = reinterpret_cast<const char*>(str);
    char ch;
    while ((ch = (char)pgm_read_byte(p++)) != '\0') write((uint8_t)ch);
  }

  void print(const char* str) {
    if (!str) return;
    while (*str) write((uint8_t)*str++);
  }

  void print(char value) { write((uint8_t)value); }
  void print(signed char value, int base = DEC) { print((long)value, base); }
  void print(unsigned char value, int base = DEC) { print((unsigned long)value, base); }
  void print(int value, int base = DEC) { print((long)value, base); }
  void print(unsigned int value, int base = DEC) { print((unsigned long)value, base); }

  void print(long value, int base = DEC) {
    if (base == DEC && value < 0) {
      write('-');
      // Handles LONG_MIN without signed overflow.
      print((unsigned long)(-(value + 1L)) + 1UL, base);
    } else {
      print((unsigned long)value, base);
    }
  }

  void print(unsigned long value, int base = DEC) {
    if (base < 2 || base > 16) base = DEC;
    char digits[11]; // 10 decimal digits for 32-bit value + spare
    uint8_t count = 0;
    do {
      const uint8_t digit = (uint8_t)(value % (unsigned)base);
      digits[count++] = (digit < 10) ? ('0' + digit) : ('A' + digit - 10);
      value /= (unsigned)base;
    } while (value && count < sizeof(digits));
    while (count) write((uint8_t)digits[--count]);
  }

  void println() { write('\r'); write('\n'); }

  template <typename T>
  void println(T value) { print(value); println(); }

  template <typename T>
  void println(T value, int base) { print(value, base); println(); }
};

// One UART instance, local to the generated application translation unit.
static R15BareUart r15_uart;
