//const client = new Paho.MQTT.Client("localhost", 9001, "web_client");
//client.connect({onSuccess: onConnect});

const client = new Paho.MQTT.Client("localhost", 8080, "web_client");
client.connect({onSuccess: onConnect});

function onConnect() {
   console.log("Connected to MQTT broker");
}


const upButton = document.getElementById("up-button");
const downButton = document.getElementById('down-button');
const leftButton = document.getElementById('left-button');
const rightButton = document.getElementById('right-button');
const shootButton = document.getElementById('shoot-button');
const stopButton = document.getElementById('stop-button');

upButton.addEventListener("click", () => {
  client.publish('robot/move', 'up');
});

downButton.addEventListener('click', () => {
  client.publish('robot/move', 'down');
});

leftButton.addEventListener('click', () => {
  client.publish('robot/move', 'left');
});

rightButton.addEventListener('click', () => {
  client.publish('robot/move', 'right');
});

shootButton.addEventListener('click', () => {
  client.publish('robot/shoot', 'fire');
});

stopButton.addEventListener('click', () => {
  // Stop the automatic robot control
});

const listItems = document.querySelectorAll('#object-list li');

listItems.forEach(item => {
  item.addEventListener('click', () => {
    const selectedObject = item.getAttribute('data-object');
    const x = item.dataset.x;
    const y = item.dataset.y;
    alert(`You selected ${selectedObject}`);
    automaticAiming(x, y);
  });
});

function automaticAiming(x, y) {
  // Each List item has some specific coordinates
  // After the user has clicked on a list item we automatically move the robot to those coordinates
  // Important, this automatic control stops as soon as the stop button is pressed
  const x_center = 300;
  const y_center = 300;
  const delta = 5; //How precise the robot tries to hit the desired coordinates

    //do something until the stop button is pressed
}