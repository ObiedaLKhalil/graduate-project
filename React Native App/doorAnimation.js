import { useRef } from "react";
import { Animated } from "react-native";

// Hook to initialize the bounceValue and animation functions
export const useDoorAnimation = () => {
  const bounceValue = useRef(new Animated.Value(1)).current;

  const openDoorAnimation = () => {
    Animated.sequence([
      Animated.timing(bounceValue, {
        toValue: 0.5, // Scale down
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 1, // Back to normal
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 1.5, // Scale up
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 2, // Scale to larger
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const closeDoorAnimation = () => {
    Animated.sequence([
      Animated.timing(bounceValue, {
        toValue: .5, // Scale down
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 1, // Reduce size
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 1.5, // Back to normal
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(bounceValue, {
        toValue: 2, // Scale smaller
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();
  };

  return { bounceValue, openDoorAnimation, closeDoorAnimation };
};
