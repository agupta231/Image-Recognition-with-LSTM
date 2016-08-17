#include <Stepper.h>
#include <NewPing.h>

#define TRIGGER_PIN 6
#define ECHO_PIN    7
#define MAX_DISTANCE 400

const int PWM_A = 3;
const int DIR_A = 12;
const int BRAKE_A = 9;
const int SNS_A = A0;

const int PWM_B = 11;
const int DIR_B = 13;
const int BRAKE_B = 8;
const int SNS_B = A1;

const int period = 50;

long time_count = 0L
int motors[] = {0, 0};

NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

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

  pinMode(BRAKE_A, OUTPUT);
  pinMode(BRAKE_B, OUTPUT);
  pinMode(DIR_A, OUTPUT);
  pinMode(DIR_B, OUTPUT);
}

void loop() {  
  if(Serial.available()) {    
    parseSerial(Serial.readString(), serial_input);

    int process = serial_input[0]
    int leftMotorPower = serial_input[1];
    int rightMotorPower = -serial_input[2];

    if(process == 0) {
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
  }

  Serial.print(time_count);
  Serial.print(":");
  Serial.println(sonar.ping_cm());
  
  time_count += period
  delay(period);
}
