// Copyright (c) 2025 Salvador E. Tropea
// Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
// License: GPLv3
// Project: ComfyUI-AudioSeparation

// This script adds an event named "set-audioseparation-toast"
// Used to notify the user in the GUI using the Toast API

import { app } from "/scripts/app.js";

// Register a new extension
app.registerExtension({
    name: "SET.AudioSeparation.ToastHandler",  // Unique name

    // The setup function is executed when the extension is loaded
    setup() {
        // Add a listener for our custom event
        app.api.addEventListener("set-audioseparation-toast", (event) => {
            // The data from Python is in event.detail
            const { message, summary, severity } = event.detail;

            // Use the ComfyUI toast API to show the message
            // app.ui.toast.addMessage is the modern way to do this
            app.extensionManager.toast.add({
                severity: severity,
                summary: summary,
                detail: message,
                life: 6000
            });
        });
    },
});
