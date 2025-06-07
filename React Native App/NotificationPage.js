import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Button, StyleSheet,Pressable } from 'react-native';
import { getStoredNotifications, clearNotifications } from './NotificationService';
import Icon from "react-native-vector-icons/Ionicons";  // You can replace this with any icon set
const NotificationPage = ({ navigation }) => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    loadNotifications();
  }, []);

  const loadNotifications = async () => {
    const storedNotifications = await getStoredNotifications();
    setNotifications(storedNotifications);
  };

  const handleClearNotifications = async () => {
    await clearNotifications();
    setNotifications([]);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Notifications</Text>
      <FlatList
        data={notifications}
        keyExtractor={(item, index) => index.toString()}
        renderItem={({ item }) => (
          <View style={styles.notificationItem}>
            <Text style={styles.title}>{item.title}</Text>
            <Text style={styles.body}>{item.body}</Text>
            <Text style={styles.time}>{item.time}</Text>
          </View>
        )}
      />

       <Pressable
            style={({ pressed }) => ({
              flexDirection: "row",
              backgroundColor: pressed ? "red" : "green",
              padding: 10,
              borderRadius: 8,
              alignItems: "center",
              opacity: pressed ? 0.7 : 1, // Add press effect
            })}
            onPress={handleClearNotifications}
          >
            
            <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
        <Text style={{ color: "#fff", fontSize: 16 }}>Clear Notifications </Text>
        <Icon name="trash-outline" size={20} color="white" />
           </View>
          </Pressable>
      
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  header: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  notificationItem: {
    padding: 10,
    marginBottom: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 5,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  body: {
    fontSize: 14,
  },
  time: {
    fontSize: 12,
    color: 'gray',
  },
});

export default NotificationPage;
