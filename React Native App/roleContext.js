import React, { createContext, useState } from 'react';

// Create Context
export const RoleContext = createContext();

export const RoleProvider = ({ children }) => {
  // Initialize with an image URL
  const [rolecont, setrolecont] = useState("monitor");

  return (
    <RoleContext.Provider value={{ rolecont, setrolecont }}>
      {children}
    </RoleContext.Provider>
  );
};
