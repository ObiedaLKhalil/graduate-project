import React ,{ useEffect, useState,useContext }  from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from "react-native-vector-icons/Ionicons";  // You can replace this with any icon set
import { View, Text } from "react-native";
import SensorActuatorControlScreen from './monitorPage';  
import LoginPage from './logInPage';
import HomePage from './homePage';
import uploadImagePage from './uploadImagePage';
import NotificationPage from './NotificationPage';
import AttendancePage from './AttendancePage';
import updateUsernamePasswordPage from './updateUsernamePasswordPage';
import updateRolePage from './updateRolePage';
import RegisterPage from './registerPage';
import AdminPage from './adminPage';
import { getFCMToken, setupNotificationListeners,requestUserPermission } from './NotificationService';
import { ImageContext,ImageProvider } from './ImageContext'; // Import Context
import { RoleContext,RoleProvider } from './roleContext'; // Import Context


//import NotificationPage from './NotificationPage';

const NotificationHandler = () => {
  const { image,setImage } = useContext(ImageContext);

  useEffect(() => {
    requestUserPermission();
    getFCMToken();
    setupNotificationListeners();
    console.log("hi");
    //setupNotificationListeners(setImage);
  }, [setImage]);

  return null;
};

const Tab = createBottomTabNavigator();
const Pages = () => {

  return (
<RoleProvider>
    <ImageProvider>
      <NotificationHandler />
    <NavigationContainer>
    <Tab.Navigator initialRouteName="Home Page">
    <Tab.Screen
  name="Register Page"
  component={RegisterPage}
  options={{ tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Register",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="create-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="create-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Register Page</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>


<Tab.Screen
  name="Update Username Password Page"
  component={updateUsernamePasswordPage}
  options={{ tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Update U_P",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="sync-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="sync-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Update Username Password Page</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>

<Tab.Screen
  name="Update Role Page"
  component={updateRolePage}
  options={{tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Update R",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="sync-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="sync-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Update Role Page</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
    <Tab.Screen
  name="Log In Page"
  component={LoginPage}
  options={{
    tabBarLabel: "Log In",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="log-in-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="log-in-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Log In</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>

<Tab.Screen
  name="Attendance Page"
  component={AttendancePage}
  options={{tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Attendance",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="calendar-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="calendar-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Attendance Page</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
    



<Tab.Screen
  name="Home Page"
  component={HomePage}
  options={{
    tabBarLabel: "Home",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="home-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="home-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Home Page</Text> 
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
     
      <Tab.Screen
  name="uploadImage Page"
  component={uploadImagePage}
  options={{tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "uploadImage",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="cloud-upload-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="cloud-upload-outline" size={24} color="black" /> 
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>uploadImage Page</Text>  
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>

   

<Tab.Screen
  name="Monitor Page"
  component={SensorActuatorControlScreen}
  options={{tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Monitor Page",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="person-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="person-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Monitor Page</Text>  
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
   



   
<Tab.Screen
  name="Admin Page"
  component={AdminPage}
  options={{tabBarButton: () => null,
  headerShown: true,               // Keep top header (optional)
    tabBarLabel: "Admin Page",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="person-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="person-outline" size={24} color="black" />  
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Admin Page</Text>  
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
   <Tab.Screen
  name="Notification Page"
  component={NotificationPage}
  options={{
    tabBarLabel: "Notification",  // This will be your page name (tab bar label)
    tabBarIcon: ({ color, size }) => (
      <Icon name="notifications-outline" size={16} color={color} />  // Add the icon in tab bar
    ),
    headerTitle: () => (
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
        <Icon name="notifications-outline" size={24} color="black" /> 
        <Text style={{ marginLeft: 8, fontSize: 18, fontWeight: 'bold' }}>Notification</Text>  
      </View>
    ),
    headerTitleAlign: 'center',  // Center the header title
  }}
/>
     
     
    </Tab.Navigator>
  </NavigationContainer>
  </ImageProvider>
  </RoleProvider>

  
  );
};

export default Pages;
