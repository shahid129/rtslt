// start the video
function startVideo() {
    const video = document.getElementById("video");
    const overlay = document.getElementById("paused-overlay");
    video.src = document.getElementById("urls").dataset.videoFeed;
    overlay.classList.add("visually-hidden");
    updateStatus("Video started");
}

// stop the video
function stopVideo() {
    const video = document.getElementById("video");
    const overlay = document.getElementById("paused-overlay");
    const placeholderImage = document.getElementById("urls").dataset.placeholderImage;
    video.src = placeholderImage;
    overlay.classList.remove("visually-hidden");
    updateStatus("Video paused");
}

// resst tje messages
function resetMessage() {
    const resetMessageUrl = document.getElementById("urls").dataset.resetMessage;
    fetch(resetMessageUrl).then(() => {
        document.getElementById("message").innerText = "";
        updateStatus("Message reset");
    });
}

// start messages
function startMessage() {
    const startMessageUrl = document.getElementById("urls").dataset.startMessage;
    fetch(startMessageUrl).then(() => {
        updateStatus("Messaging started");
    });
}

// stop or pause messages
function stopMessage() {
    const stopMessageUrl = document.getElementById("urls").dataset.stopMessage;
    fetch(stopMessageUrl).then(() => {
        updateStatus("Messaging stopped");
    });
}

// update the status whenever any button is pressed
function updateStatus(statusText) {
    document.getElementById("status").innerText = statusText;
}

// Update detected letter and message periodically
document.addEventListener("DOMContentLoaded", () => {
    setInterval(() => {
        const getDetectedLetterUrl = document.getElementById("urls").dataset.getDetectedLetter;
        fetch(getDetectedLetterUrl)
            .then(response => response.json())
            .then(data => {
                document.getElementById("detected-letter").innerText = data.letter || "Not detected";
            });
    }, 1000);

    setInterval(() => {
        const getMessageUrl = document.getElementById("urls").dataset.getMessage;
        fetch(getMessageUrl)
            .then(response => response.json())
            .then(data => {
                document.getElementById("message").innerText = data.message || "";
            });
    }, 1000);
});
