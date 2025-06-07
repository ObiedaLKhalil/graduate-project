import messaging from '@react-native-firebase/messaging';
import { Alert } from 'react-native';
import React, { useContext } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Function to store notifications in AsyncStorage
const storeNotification = async (notification) => {
  try {
    const existingNotifications = await AsyncStorage.getItem('notifications');
    const notifications = existingNotifications ? JSON.parse(existingNotifications) : [];
    notifications.unshift(notification); // Add new notification to the top
    await AsyncStorage.setItem('notifications', JSON.stringify(notifications));
  } catch (error) {
    console.error('Error saving notification:', error);
  }
};

// Request permissions for notifications
export async function requestUserPermission() {
  const authStatus = await messaging().requestPermission();
  const enabled =
    authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
    authStatus === messaging.AuthorizationStatus.PROVISIONAL;

  if (enabled) {
    console.log('Notification permission granted.');
  }
}

export async function getFCMToken() {
  const token = await messaging().getToken();
  console.log("FCM Token:", token);
  fetch('http://192.168.100.152:5001/save-token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  return token;
}

// Handle background and foreground messages
export function setupNotificationListeners() {
  console.log("I ")
  messaging().onMessage(async (remoteMessage) => {
    const notification = {
      title: remoteMessage.notification?.title || "New Notification",
      body: remoteMessage.notification?.body || "You have a new message",
      time: new Date().toLocaleString(),
    };
    console.log(remoteMessage.data.image_url);
    if (remoteMessage.data.image_url) {
      //setImage(remoteMessage.data.image_url); // Save the image in context
      }
    Alert.alert(notification.title, notification.body);
    storeNotification(notification); // Store notification in AsyncStorage
    console.log("amO");
    console.log("hi",image);
  });

  messaging().onNotificationOpenedApp((remoteMessage) => {
    console.log('Notification opened:', remoteMessage);
    console.log(remoteMessage.data.image_url);
    if (remoteMessage.data.image_url) {
     //setImage(remoteMessage.data.image_url); // Save the image in context
     console.log("obieda");


  }
  });

  messaging()
    .getInitialNotification()
    .then((remoteMessage) => {
      if (remoteMessage) {
        console.log('Notification caused app to open from quit state:', remoteMessage);
        if (remoteMessage.data.image_url) {
          //setImage(remoteMessage.data.image_url); // Save the image in context
       }
      }
    });
}

// Function to get stored notifications
export const getStoredNotifications = async () => {
  try {
    const notifications = await AsyncStorage.getItem('notifications');
    return notifications ? JSON.parse(notifications) : [];
  } catch (error) {
    console.error('Error retrieving notifications:', error);
    return [];
  }
};

// Function to clear notifications
export const clearNotifications = async () => {
  try {
    await AsyncStorage.removeItem('notifications');
  } catch (error) {
    console.error('Error clearing notifications:', error);
  }
};
