import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property var pythonBridge
    property alias prompt: promptInput.text
    property alias secret: secretInput.text
    property alias password: passwordInput.text
    signal logMessage(string message)

    radius: 10
    color: "#eef7ee"
    border.color: "#2e7d32"
    border.width: 2

    Column {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        Text {
            text: "Standard Settings"
            font.pixelSize: 16
            font.bold: true
            color: "#1b5e20"
        }

        Row {
            spacing: 10

            Column {
                spacing: 4
                Text {
                    text: "Prompt"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: promptInput
                    width: 260
                    placeholderText: "Enter prompt..."
                    text: "Steganography "
                }
            }

            Column {
                spacing: 4
                Text {
                    text: "Secret"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: secretInput
                    width: 180
                    placeholderText: "Enter secret..."
                    text: "secret"
                }
            }

            Column {
                spacing: 4
                Text {
                    text: "Password"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: passwordInput
                    width: 160
                    placeholderText: "Enter password..."
                    text: "password"
                }
            }
        }

        Row {
            spacing: 10

            Column {
                spacing: 4
                Text {
                    text: "Threshold"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: thresholdInput
                    width: 140
                    placeholderText: "0.01"
                    text: "0.01"
                }
            }

            Column {
                spacing: 4
                Text {
                    text: "EOS Threshold"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: eosThresholdInput
                    width: 160
                    placeholderText: "0.01"
                    text: "0.01"
                }
            }

            Column {
                spacing: 4
                Text {
                    text: "Top N"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                TextField {
                    id: topNInput
                    width: 120
                    placeholderText: "15"
                    text: "15"
                }
            }

            Column {
                spacing: 4
                Text {
                    text: "Actions"
                    font.pixelSize: 12
                    color: "#1b5e20"
                }
                Button {
                    text: "Apply Settings"
                    onClicked: {
                        if (!root.pythonBridge) {
                            root.logMessage("Error: pythonBridge is not available.\n")
                            return
                        }
                        var msg = root.pythonBridge.updateSettings(
                            thresholdInput.text,
                            eosThresholdInput.text,
                            topNInput.text
                        )
                        root.logMessage(msg + "\n")
                    }
                }
            }
        }
    }
}
