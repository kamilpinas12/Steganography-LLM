import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "components"

ApplicationWindow {
    id: root
    property var bridgeObject: pythonBridge

    visible: true
    width: 1800
    height: 900
    title: "LLM Steganography"
    color: "#f0f2f5"
    Component.onCompleted: showFullScreen()

    Connections {
        target: root.bridgeObject
        function onLogOutput(message) {
            terminalPanel.appendLog(message)
        }
    }

    Column {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20

        Text {
            text: "LLM Steganography Application"
            font.pixelSize: 32
            font.bold: true
            color: "#2c3e50"
            anchors.horizontalCenter: parent.horizontalCenter
        }

        Row {
            spacing: 20
            width: parent.width
            height: parent.height - 80

            Column {
                width: parent.width - terminalPanel.width - 20
                height: parent.height

                RuntimeSettingsPanel {
                    id: runtimeSettingsPanel
                    width: parent.width
                    height: 210
                    pythonBridge: root.bridgeObject
                    function onLogMessage(message) {
                        terminalPanel.appendLog(message)
                    }
                }

                Row {
                    spacing: 20
                    width: parent.width

                    Column {
                        width: (parent.width - 20) / 2
                        spacing: 12

                        EncodePanel {
                            width: parent.width
                            height: 540
                            pythonBridge: root.bridgeObject
                            prompt: runtimeSettingsPanel.prompt
                            secret: runtimeSettingsPanel.secret
                            password: runtimeSettingsPanel.password
                        }
                    }

                    Column {
                        width: (parent.width - 20) / 2
                        spacing: 12

                        DecodePanel {
                            width: parent.width
                            height: 540
                            pythonBridge: root.bridgeObject
                            prompt: runtimeSettingsPanel.prompt
                            password: runtimeSettingsPanel.password
                        }
                    }
                }
            }

            TerminalPanel {
                id: terminalPanel
                width: 1100
                height: parent.height
            }
        }
    }
}
