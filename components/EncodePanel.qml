import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property var pythonBridge
    property string prompt: ""
    property string secret: ""
    property string password: ""

    radius: 10
    color: "#e8f4f8"
    border.color: "#3498db"
    border.width: 2

    Column {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 12

        Text {
            text: "Encode"
            font.pixelSize: 18
            font.bold: true
            color: "#2980b9"
        }

        Button {
            text: "Generate & Encode"
            width: parent.width
            onClicked: {
                if (!root.pythonBridge) {
                    resultText.text = "Error: pythonBridge is not available."
                    return
                }
                resultText.text = root.pythonBridge.generateText(root.prompt, root.secret, root.password)
            }
        }

        Text {
            text: "Generated Text"
            font.pixelSize: 11
            color: "#34495e"
            font.bold: true
        }

        Rectangle {
            width: parent.width
            height: parent.height - 90
            color: "#ffffff"
            border.color: "#27ae60"
            border.width: 1
            radius: 5

            TextEdit {
                id: resultText
                anchors.fill: parent
                anchors.margins: 8
                wrapMode: TextEdit.Wrap
                readOnly: true
                color: '#000000'
                font.pixelSize: 18
            }
        }
    }
}
