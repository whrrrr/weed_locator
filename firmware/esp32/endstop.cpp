#include "endstop.h"
#include <Arduino.h>

Endstop::Endstop(int a_min_pin, int a_dir_pin, int a_step_pin, int a_en_pin, int a_switch_input, int a_step_offset, int a_home_dwell){
  min_pin = a_min_pin;
  dir_pin = a_dir_pin;
  step_pin = a_step_pin;
  en_pin = a_en_pin;
  switch_input = a_switch_input;
  home_dwell = a_home_dwell;
  step_offset = a_step_offset;
  pinMode(min_pin, INPUT_PULLUP);
}

void Endstop::home(bool dir) {
  digitalWrite(en_pin, LOW);
  delayMicroseconds(5);
  if (dir==1){
    digitalWrite(dir_pin, HIGH);
  }
  else {
    digitalWrite(dir_pin, LOW);
  }
  delayMicroseconds(5);
  bState = digitalRead(min_pin);
  while (bState != switch_input) {
    digitalWrite(step_pin, HIGH);
    digitalWrite(step_pin, LOW);
    delayMicroseconds(home_dwell);
    bState = digitalRead(min_pin);
  }
  homeOffset(dir);
}

void Endstop::homeOffset(bool dir){
  if (dir==1){
    digitalWrite(dir_pin, LOW);
  }
  else{
    digitalWrite(dir_pin, HIGH);
  }
  delayMicroseconds(5);
  for (int i = 1; i <= step_offset; i++) {
    digitalWrite(step_pin, HIGH);
    digitalWrite(step_pin, LOW);
    delayMicroseconds(home_dwell);
  }
}

bool Endstop::state(){
  bState = digitalRead(min_pin);
  return bState;
}

void Endstop::homeAll(Endstop& axis1, Endstop& axis2, Endstop& axis3, bool dir) {
  // 启用电机
  digitalWrite(axis1.en_pin, LOW);
  digitalWrite(axis2.en_pin, LOW);
  digitalWrite(axis3.en_pin, LOW);
  delayMicroseconds(5);

  // 设置方向
  digitalWrite(axis1.dir_pin, dir ? HIGH : LOW);
  digitalWrite(axis2.dir_pin, dir ? HIGH : LOW);
  digitalWrite(axis3.dir_pin, dir ? HIGH : LOW);
  delayMicroseconds(5);

  // 循环直到三个轴都触发
  bool done1 = false, done2 = false, done3 = false;

  while (!(done1 && done2 && done3)) {
    if (!done1 && digitalRead(axis1.min_pin) != axis1.switch_input) {
      digitalWrite(axis1.step_pin, HIGH);
      digitalWrite(axis1.step_pin, LOW);
    } else {
      done1 = true;
    }

    if (!done2 && digitalRead(axis2.min_pin) != axis2.switch_input) {
      digitalWrite(axis2.step_pin, HIGH);
      digitalWrite(axis2.step_pin, LOW);
    } else {
      done2 = true;
    }

    if (!done3 && digitalRead(axis3.min_pin) != axis3.switch_input) {
      digitalWrite(axis3.step_pin, HIGH);
      digitalWrite(axis3.step_pin, LOW);
    } else {
      done3 = true;
    }

    delayMicroseconds(axis1.home_dwell);  // 假设 dwell 一样
  }

  // 执行归零后的 offset
  axis1.homeOffset(dir);
  axis2.homeOffset(dir);
  axis3.homeOffset(dir);
}
