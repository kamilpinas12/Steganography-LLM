import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property var pythonBridge
    property string prompt: ""
    property string password: ""

    radius: 10
    color: "#fdf0e8"
    border.color: "#e74c3c"
    border.width: 2

    Column {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        Text {
            text: "Decode"
            font.pixelSize: 18
            font.bold: true
            color: "#c0392b"
        }

        Button {
            text: "Decode Secret"
            width: parent.width
            onClicked: {
                if (!root.pythonBridge) {
                    resultText.text = "Error: pythonBridge is not available."
                    return
                }
                resultText.text = root.pythonBridge.generateSecret(root.prompt, root.password)
            }
        }

        Text {
            text: "Decoded Secret"
            font.pixelSize: 11
            color: "#34495e"
            font.bold: true
        }

        Rectangle {
            width: parent.width
            height: parent.height - 90
            color: "#ffffff"
            border.color: "#f39c12"
            border.width: 1
            radius: 5

            TextEdit {
                id: resultText
                anchors.fill: parent
                anchors.margins: 8
                wrapMode: TextEdit.Wrap
                readOnly: true
                selectByMouse: true
                color: '#000000'
                font.pixelSize: 18
            }
        }
    }
}
