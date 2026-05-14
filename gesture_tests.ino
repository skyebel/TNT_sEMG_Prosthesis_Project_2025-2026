#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver board = Adafruit_PWMServoDriver(0x40);
#define SERVOMIN 125
#define SERVOMAX 1250
int angleToPulse(int ang) {
  return map(ang, 0, 360, SERVOMIN, SERVOMAX);
}
void move_hand(int thumb, int index, int middle, int ring, int pinkie) {
  board.setPWM(0, 0, angleToPulse(thumb));
  delay(100);
  board.setPWM(1, 0, angleToPulse(index));
  delay(100);
  board.setPWM(2, 0, angleToPulse(middle));
  delay(100);
  board.setPWM(3, 0, angleToPulse(ring));
  delay(100);
  board.setPWM(4, 0, angleToPulse(pinkie));
  delay(100);
}
void releaseAll() {
  for (uint8_t ch = 0; ch < 5; ch++) {
    board.setPWM(ch, 0, 0);
  }
}
// Gesture definitions {thumb, index, middle, ring, pinkie}
const char* GESTURE_NAMES[] = {"RELAX", "FIST", "ROCK", "PEACE", "SHAKA"};
const uint8_t NUM_GESTURES = 5;
void applyGesture(int i) {
  switch (i) {
    case 0: move_hand(110,   170,   170, 170,   170); break; // RELAX
    case 1: move_hand(160, 0, 0, 0, 0); break; // FIST
    case 2: move_hand(160,   170,   0, 0,   170); break; // ROCK
    case 3: move_hand(160,   170, 170, 0, 0); break; // PEACE
    case 4: move_hand(110, 0, 0, 0,   170); break; // SHAKA
  }
}
void setup() {
  Serial.begin(9600);
  Wire.begin();
  board.begin();
  board.setPWMFreq(60);
  delay(500);
  Serial.println("BOOT");
  move_hand(110,   170,   170, 170,   170);
  delay(2000);
  Serial.println("READY");
}
void loop() {
  for (uint8_t i = 0; i < NUM_GESTURES; i++) {
    Serial.print("Gesture: ");
    Serial.println(GESTURE_NAMES[i]);
    applyGesture(i);
    delay(3000);       // hold gesture
    Serial.println("Rest");
    move_hand(110,   170,   170, 170,   170);
    delay(500);
    releaseAll();
    delay(2000);       // rest between gestures
  }
}
