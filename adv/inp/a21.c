#include <stdio.h>
#include <stdbool.h>

int main(void) {
  long long reg0 = 0LL;
  long long reg1 = 0LL;
  long long reg2 = 0LL;
  long long reg3 = 0LL;
  long long reg4 = 0LL;
  long long reg5 = 0LL;

  bool done = false;
  while (!done) {
    switch (reg5) {
    case 0:
      reg3 = 123LL;
      reg5++;

    case 1:
      reg3 = reg3 & 456LL;
      reg5++;

    case 2:
      reg3 = reg3 == 72LL;
      reg5++;

    case 3:
      reg5 = reg3 + reg5;
      break;

    case 4:
      reg5 = 0LL;
      break;

    case 5:
      reg3 = 0LL;
      reg5++;

    case 6:
      reg2 = reg3 | 65536LL;
      reg5++;

    case 7:
      reg3 = 832312LL;
      reg5++;

    case 8:
      reg1 = reg2 & 255LL;
      reg5++;

    case 9:
      reg3 = reg3 + reg1;
      reg5++;

    case 10:
      reg3 = reg3 & 16777215LL;
      reg5++;

    case 11:
      reg3 = reg3 * 65899LL;
      reg5++;

    case 12:
      reg3 = reg3 & 16777215LL;
      reg5++;

    case 13:
      reg1 = 256LL > reg2;
      reg5++;

    case 14:
      reg5 = reg1 + reg5;
      break;

    case 15:
      reg5 = reg5 + 1LL;
      break;

    case 16:
      reg5 = 27LL;
      break;

    case 17:
      reg1 = 0LL;
      reg5++;

    case 18:
      reg4 = reg1 + 1LL;
      reg5++;

    case 19:
      reg4 = reg4 * 256LL;
      reg5++;

    case 20:
      reg4 = reg4 > reg2;
      reg5++;

    case 21:
      reg5 = reg4 + reg5;
      break;

    case 22:
      reg5 = reg5 + 1LL;
      break;

    case 23:
      reg5 = 25LL;
      break;

    case 24:
      reg1 = reg1 + 1LL;
      reg5++;

    case 25:
      reg5 = 17LL;
      break;

    case 26:
      reg2 = reg1;
      reg5++;

    case 27:
      reg5 = 7LL;
      break;

    case 28:
      reg1 = reg3 == reg0;
      reg5++;

    case 29:
      reg5 = reg1 + reg5;
      break;

    case 30:
      reg5 = 5LL;
      break;

    default:
      printf("Out of range: %lld\n", reg5);
      done = true;
    }
    reg5++;
  }
  return 0;
}