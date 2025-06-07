import React, { useState,useRef  } from 'react';
import { View, ScrollView,Text, Button, StyleSheet, Image,Pressable } from 'react-native';
import Icon from "react-native-vector-icons/Ionicons";  // You can replace this with any icon set
const AdminPage = ({ navigation }) => {
    const scrollViewRef = useRef<ScrollView>(null);
const [adminPageImageUri, setadminPageImageUri] = useState(require("./admin.png"));
    return (
        <ScrollView contentContainerStyle={styles.container}>   
           {adminPageImageUri && <Image source={adminPageImageUri} style={styles.image}  />}
   
      <View style={styles.button}>
        <Button title="Monitor Page" onPress={() => navigation.navigate('Monitor Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Register Page" onPress={() => navigation.navigate('Register Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Update Username Password Page" onPress={() => navigation.navigate('Update Username Password Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Update Role Page" onPress={() => navigation.navigate('Update Role Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Log In Page" onPress={() => navigation.navigate('Log In Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Attendance Page" onPress={() => navigation.navigate('Attendance Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Home Page" onPress={() => navigation.navigate('Home Page')} color="#304548" />
      </View>

      <View style={styles.button}>
        <Button title="uploadImage Page" onPress={() => navigation.navigate('uploadImage Page')} color="#304548" />
      </View>
      <View style={styles.button}>
        <Button title="Notification Page" onPress={() => navigation.navigate('Notification Page')} color="#304548" />
      </View>
      
      </ScrollView>


    );


};
const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    flexGrow: 1,

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
  image: {

        width: 100,
        height: 100,
       
   
   
  },
 
});

export default AdminPage;
