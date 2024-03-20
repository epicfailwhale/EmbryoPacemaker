
/*!
* @file HeartRateMonitor.ino
* @brief HeartRateMonitor.ino  Sampling and ECG output
*
*  Real-time sampling and ECG output
*
* @author linfeng(490289303@qq.com)
* @version  V1.0
* @date  2016-4-5
*/
const int heartPin = A0;
const int outputPin = 13;
void setup() {
  Serial.begin(115200);
  pinMode(outputPin, OUTPUT);
}
void loop() {
    if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the incoming string until newline

    if (command == "TRIG ON") {
      digitalWrite(outputPin, HIGH); // Set the output pin HIGH (5V)
    } else if (command == "TRIG OFF") {
      digitalWrite(outputPin, LOW); // Set the output pin LOW (0V)
    }
  }
  int heartValue = analogRead(heartPin);
  Serial.println(heartValue);
  delay(5);
}