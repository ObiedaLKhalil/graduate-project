import React, { useRef } from "react";
import { View, StyleSheet, Animated, Button } from "react-native";

const SmallImageAnimation = () => {
  const bounceValue = useRef(new Animated.Value(1)).current;

  // Animation Function
  const startBounce = () => {
    Animated.sequence([
      Animated.timing(bounceValue, {
        toValue: 1.5, // Scale up
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 1, // Scale down back to normal
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  };

  return (
    <View style={styles.container}>
      <Animated.Image
        source={require('./openDoor.png')} // Update to your local asset path
        style={[
          styles.image,
          {
            transform: [{ scale: bounceValue }], // Bind scale animation
          },
        ]}
      />
      <Button title="Animate Image" onPress={startBounce} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
  },
  image: {
    width: 100, // Small image width
    height: 50, // Small image height
   // borderRadius: 25, // Optional: Make it circular
  },
});

export default SmallImageAnimation;
