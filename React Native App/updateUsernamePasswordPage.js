import React, { useState,useContext } from "react";
import { View, Text, TextInput, Button, Alert, StyleSheet } from "react-native";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { RoleContext } from './roleContext';
const updateUsernamePasswordPage = ({ navigation }) => {
  const [oldUsername, setOldUsername] = useState("");
  const [oldPassword, setOldPassword] = useState("");
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const { rolecont, setrolecont } = useContext(RoleContext);

  const handleUpdate = async () => {
    try {
      const response = await axios.put("http://192.168.100.152:5001/update", {
        old_username: oldUsername,
        old_password: oldPassword,
        new_username: newUsername || null,
        new_password: newPassword || null,
      });

      if (response.data.status === "success") {
        Alert.alert("Update Successful", "Your credentials have been updated.");
        await AsyncStorage.setItem("username", newUsername || oldUsername);
          navigation.navigate("Log In Page");   
      }
    } catch (error) {
      Alert.alert("Update Failed", "Please check your details and try again.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Update Username and Password</Text>
      <TextInput
        placeholder="Old Username"
        value={oldUsername}
        onChangeText={setOldUsername}
        style={styles.input}
      />
       <TextInput
        placeholder="Old Password"
        value={oldPassword}
        onChangeText={setOldPassword}
        style={styles.input}
      />
      <TextInput
        placeholder="New Username (optional)"
        value={newUsername}
        onChangeText={setNewUsername}
        style={styles.input}
      />
      <TextInput
        placeholder="New Password (optional)"
        value={newPassword}
        onChangeText={setNewPassword}
        style={styles.input}
        secureTextEntry
      />
      <Button title="Update" onPress={handleUpdate} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 20 },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginBottom: 20 },
  input: { borderBottomWidth: 1, marginBottom: 10, padding: 8 },
});

export default updateUsernamePasswordPage;
