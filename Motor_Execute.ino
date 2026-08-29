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

void applyRest() {
  move_hand(110, 170, 170, 170, 170);
}

void applyGesture(int gesture_id) {
  switch (gesture_id) {
    case 0: applyRest();                                        Serial.println("ACK:RELAX");  break;
    case 1: move_hand(160,   0,   0,   0,   0);                Serial.println("ACK:FIST");   break;
    case 2: move_hand(160, 170,   0,   0, 170);                Serial.println("ACK:ROCK");   break;
    case 3: move_hand(160, 170, 170,   0,   0);                Serial.println("ACK:PEACE");  break;
    case 4: move_hand(110,   0,   0,   0, 170);                Serial.println("ACK:SHAKA");  break;
    default: applyRest();                                       Serial.println("ACK:REST");   break;
  }
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);
  Wire.begin();
  board.begin();
  board.setPWMFreq(60);
  delay(500);
  applyRest();
  delay(500);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;
    int gesture_id = (int)line.toFloat();
    applyGesture(gesture_id);
  }
}
