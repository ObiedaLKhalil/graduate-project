import React, { createContext, useState } from 'react';

// Create Context
export const ImageContext = createContext();

export const ImageProvider = ({ children }) => {
  // Initialize with an image URL
 const [image, setImage] = useState("https://roc.ai/wp-content/uploads/2018/11/howfrworks.png");

  //const [image, setImage] = useState(" http://192.168.100.152:5001/uploads/image.jpg");
  return (
    <ImageContext.Provider value={{ image, setImage }}>
      {children}
    </ImageContext.Provider>
  );
};
