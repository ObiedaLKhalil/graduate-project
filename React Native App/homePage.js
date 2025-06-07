import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, Image,Pressable } from 'react-native';
import Icon from "react-native-vector-icons/Ionicons";  // You can replace this with any icon set
const HomePage = ({ navigation }) => {
  const [homePageImageUri, sethomePageImageUri] = useState(require("./homeImage.png"));
  
  return (
    <View style={styles.container}>
      {homePageImageUri && <Image source={homePageImageUri} />}
      <View style={styles.button}>
        <Button title="Login" onPress={() => navigation.navigate('Log In Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Change the username and password" onPress={() => navigation.navigate('Update Username Password Page')} color="#304548" />
      </View>
     
      <View style={styles.container}>
      {/* Notification Button */}
      <Pressable
        style={styles.notificationButton}
        onPress={() => navigation.navigate('Notification Page')}
      >
        <Icon name="notifications-outline" size={24} color="white" />
      </Pressable>

     
    </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  button: {
    marginVertical: 10, // Add vertical space between buttons
    width: '60%', // Make buttons occupy 60% of the screen width
    alignItems: 'center', // Center text inside the button
  },
  header: {
    fontSize: 50,
    marginBottom: 20,
    color: "white",
    fontWeight: "bold",
  },
  notificationButton: {
    position: 'absolute',
    top: -435, // Adjust to align with the status bar
    right: 150, // Adjust as needed
    backgroundColor: '#304548',
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default HomePage;
