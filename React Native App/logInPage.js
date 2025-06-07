import React, { useState,useContext } from "react";
import { View, Text, TextInput, Button, Alert, StyleSheet } from "react-native";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { RoleContext } from './roleContext';
import { getFCMToken, setupNotificationListeners,requestUserPermission } from './NotificationService';
const LoginPage = ({ navigation }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { rolecont, setrolecont } = useContext(RoleContext);

  const handleLogin = async () => {
    try {
      const response = await axios.post("http://192.168.100.152:5001/login", { username:username, password:password });
      console.log(response.data.status);
      if (response.data.status === "success") {
        await AsyncStorage.setItem("userRole", response.data.role);
        await AsyncStorage.setItem("username", response.data.username);
        console.log(response.data.role);
        Alert.alert("Login Successful");
        setrolecont(response.data.role);
        if (response.data.role === "admin") {
          getFCMToken();
         
          navigation.navigate("Admin Page");
        } else {
          navigation.navigate("Monitor Page");
          getFCMToken();
        }
      }
    } catch (error) {
      console.log(error);  // Log the error
      Alert.alert("Login Failed", "Invalid username or password");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Login</Text>
      <TextInput placeholder="Username" value={username} onChangeText={setUsername} style={styles.input} />
      <TextInput placeholder="Password" value={password} onChangeText={setPassword} style={styles.input} secureTextEntry />
      <Button title="Login" onPress={handleLogin} />
      <Button title="Update Username Password Page" onPress={() => navigation.navigate("Update Username Password Page")} />

    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 20 },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginBottom: 20 },
  input: { borderBottomWidth: 1, marginBottom: 10, padding: 8 }
});

export default LoginPage;