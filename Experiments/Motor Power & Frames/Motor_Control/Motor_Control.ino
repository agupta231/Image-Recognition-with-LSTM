#include <Stepper.h>

const int PWM_A = 3;
const int DIR_A = 12;
const int BRAKE_A = 9;
const int SNS_A = A0;

const int PWM_B = 11;
const int DIR_B = 13;
const int BRAKE_B = 8;
const int SNS_B = A1;

int motors[] = {0, 0};

void parseSerial(String input, int motorValues[]) {
  char seperator = ':';
  
  int leftMotorPower = 0;
  int rightMotorPower = 0;
  
  for(int i = 0; i < input.length(); i++) {
    if(input.charAt(i) == seperator) {
      leftMotorPower = input.substring(0, i).toInt();
      rightMotorPower = input.substring(i+1, input.length()).toInt();  
    }
  }
  
  motorValues[0] = leftMotorPower;
  motorValues[1] = rightMotorPower;
}

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(20);

  pinMode(PWM_A, OUTPUT);
  pinMode(PWM_B, OUTPUT);
  pinMode(BRAKE_A, OUTPUT);
  pinMode(BRAKE_B, OUTPUT);
}

void loop() {
  if(Serial.available()) {
    parseSerial(Serial.readString(), motors);
    
    int leftMotorPower = motors[0];
    int rightMotorPower = motors[1];

    Serial.println(leftMotorPower);
    Serial.println(rightMotorPower);

    if(leftMotorPower > 0) {
      digitalWrite(BRAKE_A, LOW);
      digitalWrite(DIR_A, HIGH);

      delay(10);
    }
    else if(leftMotorPower < 0) {
      digitalWrite(BRAKE_A, LOW);
      digitalWrite(DIR_A, LOW);

      delay(10);
    }
    else if(leftMotorPower == 0) {
      digitalWrite(BRAKE_A, HIGH);
    }

    if(rightMotorPower > 0) {
      digitalWrite(BRAKE_B, LOW);
      digitalWrite(DIR_B, HIGH);

      delay(10);
    }
    else if(rightMotorPower < 0) {
      digitalWrite(BRAKE_B, LOW);
      digitalWrite(DIR_B, LOW);
    
      delay(10);
    }
    else if(rightMotorPower == 0) {
      digitalWrite(BRAKE_B, HIGH);
    }

    analogWrite(PWM_A, abs(leftMotorPower));
    analogWrite(PWM_B, abs(rightMotorPower));  
  }

  delay(50);
}
