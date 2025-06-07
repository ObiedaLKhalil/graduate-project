import React, { useState,useContext,useRef,useEffect } from "react";
import { View, Text, Button, Alert, Animated, StyleSheet ,Image,ScrollView} from "react-native";
import axios from "axios";
import { useDoorAnimation } from "./doorAnimation"; // Import the custom hook
import { ImageContext } from './ImageContext'; // Import Context
//<Button title="Receive Image" onPress={receiveImage} />
//{faceDetectImageUri && <Image source={faceDetectImageUri } style={styles.faceDetectImageImage} color="blue" />}
import RNFetchBlob from 'rn-fetch-blob'; // Import rn-fetch-blob
import RNFS from 'react-native-fs';


const SensorActuatorControlScreen = ({ navigation }) => {
  const [linear_solenoidStatus, setLinear_solenoidStatus] = useState("Close");
  const [PIRStatus, setcapture1Status] = useState("off");
  const [faceDetectImageUri, setFaceDetectImageUri] = useState(require("./faceDetectImage.png"));
  // Use the hook to get animation values and functions
  const { bounceValue, openDoorAnimation, closeDoorAnimation } =
    useDoorAnimation();
    const { image,setImage } = useContext(ImageContext);
    const [error, setError] = useState(null);

     const scrollViewRef = useRef<ScrollView>(null);
     const downloadImage = async () => {
      //const fileName = 'downloaded-image.jpg';
      const timestamp = Date.now();
      const fileName = `image_${timestamp}.jpg`;

      const localPath = `${RNFS.DocumentDirectoryPath}/${fileName}`;
  
      try {
        const downloadResult = await RNFS.downloadFile({
          fromUrl: 'http://192.168.100.152:5001/get-image', // Replace with your Pi IP
          toFile: localPath,
        }).promise;
  
        if (downloadResult.statusCode === 200) {
          setImage('file://' + localPath);
          console.log('Image saved to:', localPath);
        } else {
          setError('Failed to download image');
          console.log('Download error:', downloadResult);
        }
      } catch (err) {
        console.log('Error fetching image:', err);
        setError(err.message);
      }
    };

  const toggleLinear_solenoid = async (state) => {
    try {
      const response = await axios.post(
        `http://192.168.100.152:5001/solenoid/${state}`
      );
      if (state === "on") {
        setLinear_solenoidStatus("Open");
      } else {
        setLinear_solenoidStatus("Close");
      }
      Alert.alert("Success", response.data.status);
    } catch (error) {
      Alert.alert("Error", "Failed to control Linear_solenoid");
    }
  };
const captureFromBothCameras = async () => {
  try {
    const response = await axios.post('http://192.168.100.152:5001/capture12/');
    
    // Optional: You can update a status state if needed
    setcapture1Status('captured');

    Alert.alert('Success', response.data.status);
  } catch (error) {
    Alert.alert('Error', 'Failed to capture images');
    console.error(error);
  }
};
    
  
    

  return (
    <View style={styles.container}>
       {image ? (
        <Image source={{ uri: image }} style={styles.faceDetectImageImage} />
      ) : (
        <Text>No image available</Text>
      )}
      <Text style={styles.text}>Door is currently: {linear_solenoidStatus}</Text>

      <Image
  source={
    linear_solenoidStatus === "Open"
      ? require("./openDoor.png")
      : require("./closeDoor.png")
  }
  style={styles.image}
/>
    <View style={styles.button}>
      <Button
        title="Open the door"
        onPress={() => {
          toggleLinear_solenoid("on");
          openDoorAnimation();
        }}
        color="green"
      />
      <Button
        title="Close the door"
        onPress={() => {
          toggleLinear_solenoid("off");
          closeDoorAnimation();
        }}
        color="red"
      />
      </View>
      <Button
        title="capture image from external,internal cam"
        onPress={() => {
          captureFromBothCameras()
              }
        }
        color="orange"
      />
      <Button title="Receive Image" onPress={downloadImage} />
        </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
    //position: 'relative',
  },
  image: {
    width: 200, // Image width
    height: 100, // Image height
   
    position: 'relative',
    // borderRadius: 25, // Optional: Make it circular
  },
  button: {
    flexDirection: "row", // Arrange buttons in a horizontal row
    marginTop: 20, // Add spacing between the buttons and other content
    justifyContent: "space-between", // Space out buttons
    alignItems: "center",
    width: "60%", // Adjust width to contain buttons properly
  },
  faceDetectImageImage: {
    width: 100,
    height: 200,
  },
  text:{
  //  position: 'relative',
   // marginTop: 200,
  }
});

export default SensorActuatorControlScreen;
