
/*
  simpleMovements.ino

 This  sketch simpleMovements shows how they move each servo motor of Braccio

 Created on 18 Nov 2015
 by Andrea Martino

 This example is in the public domain.
 */

#include <Braccio.h>
#include <Servo.h>
#include <Wire.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

#define SLAVE_ADRESS 11

int M0=90;
int M1=45;
int M2=180;
int M3=90; // damit Zeiger in der Mitte 
int M4=90;
int M5=10;

int sendeInhalt =0;
int position_ud=M3;
int position_lr =M0 ;

int Shoot_pin = 13;

int coordinate_arr[2];
int counter=0;

void setup() {
  pinMode(Shoot_pin,OUTPUT);
  
  Braccio.begin();
  delay(200);
  Braccio.ServoMovement(20, M0, M1, M2, M3, M4, M5);
  delay(200);
  
  Wire.begin(SLAVE_ADRESS);
  Wire.onReceive(receiveEvents);
   
}

void loop() {
} // End loop



void receiveEvents(int numBytes)
{
  while (Wire.available()) //"wenn ein Datenpaket geliefert wird"
  {
    sendeInhalt =Wire.read();
    if(sendeInhalt==254)//Up -> M2 requierd
    {
      if (position_ud > 10)
      {
      position_ud = position_ud - 10;
      Braccio.ServoMovement(20, position_lr, M1, M2, position_ud, M4,  M5);  
      delay(200);
      Serial.println("Up");
      }
      else 
      {
        position_ud=M3;
        Braccio.ServoMovement(20, position_lr, M1, M2,position_ud, M4, M5); 
        delay(200);
      }
      
    }
    if(sendeInhalt==253)//down-> M2 requierd
    {
      if(position_ud < 170)
      {
        
      position_ud = position_ud + 10;
       Braccio.ServoMovement(20, position_lr,M1, M2, position_ud, M4, M5);   
      delay(200);
      Serial.println("down");
      }else 
      {
       position_ud=M3;
       Braccio.ServoMovement(20, position_lr, M1, M2,position_ud, M4, M5);  
       delay(200);
      }
      
    }
    if(sendeInhalt==252)//left-> M2 requierd
    {
      if(position_lr < 170)
      {
      position_lr = position_lr + 10;
       Braccio.ServoMovement(20, position_lr,M1, M2, position_ud, M4, M5);  
       delay(200);
       Serial.println("left");
      }
      else {
        position_lr=M0;
        Braccio.ServoMovement(20, position_lr, M1, M2,position_ud, M4, M5); 
        delay(200);
      }
      
    }
    if(sendeInhalt==251)//right-> M2 requierd
    {
      if (position_lr > 10)
      {
      position_lr = position_lr - 10;
      Braccio.ServoMovement(20, position_lr, M1, M2,position_ud, M4, M5);  
      delay(200);
      Serial.println("right");
      }
      else {
        position_lr=M0;
        Braccio.ServoMovement(20, position_lr, M1, M2,position_ud, M4, M5); 
        delay(200);
      }
     
    }
    if (sendeInhalt==250)// Shoot Bfehl
    { 
      digitalWrite(Shoot_pin,HIGH);
      delay(1000);
      digitalWrite(Shoot_pin,LOW);
    }
    if (sendeInhalt < 180) // Koordinatenvorgabe
    {
    if(counter==0)
    {
      coordinate_arr[counter]=sendeInhalt;
      counter++;
    }
    else if(counter==1){
      coordinate_arr[counter]=sendeInhalt;
      counter=0;
      int x = round(coordinate_arr[0] / 10.0) * 10;
      int y = round(coordinate_arr[1] / 10.0) * 10;
      if (x > 0 && x < 180 && y > 0 && y < 180) {
      Braccio.ServoMovement(20, x, M1, M2,y, M4, M5);  
      delay(200);
        }  
      }
    }
  } // End while(serial.available)
}// End receiveEvents
