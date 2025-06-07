import React, { useState } from "react";
import { View, Button, Image, Alert, PermissionsAndroid, Platform,Pressable,Text } from "react-native";
import { launchImageLibrary } from "react-native-image-picker";
import Icon from "react-native-vector-icons/Ionicons";  // You can replace this with any icon set
const SERVER_URL = "http://192.168.100.152:5001/upload"; // Replace with your Raspberry Pi's IP
//
const UploadImagePage = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  const requestGalleryPermission = async () => {
    if (Platform.OS === "android") {
      try {
        let permission =
          Platform.Version >= 33
            ? PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES
            : PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE;

        const granted = await PermissionsAndroid.request(permission);
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    }
    return true;
  };

  const selectImage = async () => {
    const hasPermission = await requestGalleryPermission();
    if (!hasPermission) {
      Alert.alert("Permission Denied", "You need to allow access to your gallery.");
      return;
    }

    launchImageLibrary({ mediaType: "photo" }, (response) => {
      if (response.didCancel) {
        console.log("User cancelled image picker");
      } else if (response.errorMessage) {
        console.log("Image Picker Error: ", response.errorMessage);
      } else if (response.assets && response.assets.length > 0) {
        setSelectedImage(response.assets[0].uri);
      }
    });
  };

  const uploadImage = async () => {
    if (!selectedImage) {
      Alert.alert("No Image Selected", "Please select an image first.");
      return;
    }
    console.log("Selected Image URI:", selectedImage);

    const formData = new FormData();
    formData.append("image", {
      uri: selectedImage,
      name: "uploaded_image.jpg",
      type: "image/jpeg",
    });
  
    try {
      console.log("FormData being sent:", formData); // Debugging

      const response = await fetch(SERVER_URL, {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
  
      // Check if the response is successful
      if (!response.ok) {
        throw new Error("Failed to upload image");
      }
  
      const result = await response.json();
  
      // Check if result has a message
      if (result && result.message) {
        Alert.alert("Upload Success", result.message);
      } else {
        Alert.alert("Upload Success", "Image uploaded successfully but no message returned.");
      }
    } catch (error) {
      console.error("Upload Error:", error);
      Alert.alert("Upload Failed", error.message || "Error uploading image.");
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
       <Pressable
      style={({ pressed }) => ({
        flexDirection: "row",
        backgroundColor: pressed ? "green" : "green",
        padding: 10,
        borderRadius: 8,
        alignItems: "center",
        opacity: pressed ? 0.7 : 1, // Add press effect
      })}
      onPress={selectImage}
    >
      <Text style={{ color: "#fff", marginLeft: 8, fontSize: 16 }}>select Image  </Text>
      <Icon name="albums-outline" size={20} color="white" />
      
    </Pressable>
      
      
      {selectedImage && (
        <>
          <Image source={{ uri: selectedImage }} style={{ width: 200, height: 200, marginTop: 20 ,marginBottom: 20,}} />
          <Pressable
      style={({ pressed }) => ({
        flexDirection: "row",
        backgroundColor: pressed ? "green" : "green",
        padding: 10,
        borderRadius: 8,
        alignItems: "center",
        opacity: pressed ? 0.7 : 1, // Add press effect
      })}
      onPress={uploadImage}
    >
      <Text style={{ color: "#fff", marginLeft: 8, fontSize: 16 }}>Upload  </Text>
      <Icon name="cloud-upload-outline" size={20} color="white" />
      
    </Pressable>
        </>
      )}
    </View>
  );
};

export default UploadImagePage;
