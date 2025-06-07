import React, { useState } from "react";
import { View, Text, TextInput, Button, Alert, StyleSheet } from "react-native";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

const updateRolePage = ({ navigation }) => {
  const [username, setUsername] = useState("");
  const [newRole, setNewRole] = useState("");

  const handleUpdateRole = async () => {
    try {
      const response = await axios.put("http://192.168.100.152:5001/update_role", {
        username,
        new_role: newRole,
      });

      if (response.data.status === "success") {
        Alert.alert("Update Successful", "Role has been updated.");
        await AsyncStorage.setItem("role", newRole);
        navigation.navigate("Admin Page");
      }
    } catch (error) {
      Alert.alert("Update Failed", "Please check your details and try again.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Update Role</Text>
      <TextInput
        placeholder="Username"
        value={username}
        onChangeText={setUsername}
        style={styles.input}
      />
      <TextInput
        placeholder="New Role (admin/monitor)"
        value={newRole}
        onChangeText={setNewRole}
        style={styles.input}
      />
      <Button title="Update Role" onPress={handleUpdateRole} />
      <Button title="Cancel" onPress={() => navigation.goBack()} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 20 },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginBottom: 20 },
  input: { borderBottomWidth: 1, marginBottom: 10, padding: 8 },
});

export default updateRolePage;
