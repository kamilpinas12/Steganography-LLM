import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    radius: 10
    color: "#000000"
    border.color: "#3f3f3f"
    border.width: 2

    function appendLog(message) {
        if (!message || message.length === 0) {
            return
        }

        terminalOutput.text += message
        if (terminalOutput.text.length > 100000) {
            terminalOutput.text = terminalOutput.text.slice(terminalOutput.text.length - 100000)
        }

        terminalOutput.cursorPosition = terminalOutput.text.length
        terminalFlick.contentY = Math.max(0, terminalFlick.contentHeight - terminalFlick.height)
    }

    function clearLog() {
        terminalOutput.text = ""
        terminalFlick.contentY = 0
    }

    Column {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Row {
            width: parent.width
            spacing: 8

            Text {
                text: "Terminal"
                font.pixelSize: 18
                font.bold: true
                color: "#ffffff"
            }

            Button {
                text: "Clear"
                onClicked: root.clearLog()
            }
        }

        Rectangle {
            width: parent.width
            height: parent.height - 40
            radius: 6
            color: "#000000"
            border.color: "#202020"
            border.width: 1

            Flickable {
                id: terminalFlick
                anchors.fill: parent
                anchors.margins: 8
                clip: true
                contentWidth: terminalOutput.paintedWidth
                contentHeight: terminalOutput.paintedHeight

                TextEdit {
                    id: terminalOutput
                    width: terminalFlick.width
                    readOnly: true
                    wrapMode: TextEdit.WrapAnywhere
                    color: "#ffffff"
                    selectionColor: "#333333"
                    selectedTextColor: "#ffffff"
                    font.family: "monospace"
                    font.pixelSize: 13
                    text: ""
                }

                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AlwaysOn
                }
                ScrollBar.horizontal: ScrollBar {
                    policy: ScrollBar.AsNeeded
                }
            }
        }
    }
}
