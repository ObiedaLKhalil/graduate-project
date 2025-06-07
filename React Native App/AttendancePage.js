import React, { useEffect, useState } from "react";
import { View, Text, FlatList, ActivityIndicator, StyleSheet,Button,Pressable } from "react-native";
import axios from "axios";

const AttendancePage = ({ navigation }) => {
  const [attendance, setAttendance] = useState([]);
  const fetchAttendance = async () => {
    try {
      const response = await axios.get("http://192.168.100.152:5001/attendance");
      setAttendance(response.data.data);
    } catch (error) {
      console.error("Error fetching attendance data:", error);
    } 
  };
  console.log(attendance);
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Attendance Records</Text>

      <Pressable style={styles.button} onPress={fetchAttendance}>
  <Text style={styles.buttonText}>Attendance Records</Text>
</Pressable>
      
      <FlatList
        data={attendance}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Text style={styles.text}>📌 Employee: {item['employee name']}</Text>
            <Text style={styles.text}>🕒 Entry Time: {item['time of entry']}</Text>
            <Text style={styles.text}>🚪 Exit Time: {item['time of exit']}</Text>
          </View>
        )}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: "#f4f4f4" },
  title: { fontSize: 24, fontWeight: "bold", textAlign: "center", marginBottom: 16 },
  card: { backgroundColor: "white", padding: 12, borderRadius: 10, marginBottom: 10, elevation: 3 },
  text: { fontSize: 16, marginBottom: 4 },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 999,
    backgroundColor: '#4f46e5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default AttendancePage;
