// File: camera_feeds.js
// Location: /static/camera_feeds.js

/**
 * Initializes drag-and-drop for camera previews and main camera slots.
 * - Each .preview-feed is draggable.
 * - Each .camera-slot in the main area can receive a dropped preview.
 *
 * Default behavior:
 *   1) On drop, the preview feed is cloned (so the original stays in Preview).
 *   2) The slot is cleared of any previous content before the new feed is appended.
 *   3) The cloned feed in the slot is no longer draggable.
 *
 * If you'd rather "move" instead of "copy," remove the clone logic and re-append
 * the original feed from the sidebar, or remove it from the preview entirely.
 */

function initCameraFeeds() {
  // 1) Find all camera preview elements in the right sidebar
  const previewFeeds = document.querySelectorAll('.preview-feed');
  
  // 2) Find all camera slots in the main area (e.g., up to 4 slots)
  const cameraSlots = document.querySelectorAll('.camera-slot');

  // 3) Make each preview feed draggable
  previewFeeds.forEach((feed) => {
    // Ensure each feed has a unique ID if not already set
    if (!feed.id) {
      feed.id = 'previewFeed_' + Math.floor(Math.random() * 100000);
    }
    feed.setAttribute('draggable', 'true');
    feed.addEventListener('dragstart', handleDragStart);
  });

  // 4) Enable drop events on each slot
  cameraSlots.forEach((slot) => {
    slot.addEventListener('dragover', handleDragOver);
    slot.addEventListener('drop', handleDrop);
  });
}

/**
 * Stores the dragged feed's ID in the dataTransfer object.
 */
function handleDragStart(e) {
  e.dataTransfer.setData('text/plain', e.target.id);
  // Optionally log or show a visual indicator
  console.debug(`[camera_feeds] Drag start for ID: ${e.target.id}`);
}

/**
 * Allows drop by preventing the default behavior.
 */
function handleDragOver(e) {
  e.preventDefault();
}

/**
 * On drop: retrieve feed ID, clone it, and append it to the slot.
 */
function handleDrop(e) {
  e.preventDefault();

  // 1) Get the feed's ID
  const feedId = e.dataTransfer.getData('text/plain');
  const feedElement = document.getElementById(feedId);
  if (!feedElement) return;

  // 2) Clear the slot (remove if you want multiple feeds per slot)
  e.currentTarget.innerHTML = '';

  // 3) Clone the feed
  const clonedFeed = feedElement.cloneNode(true);
  clonedFeed.removeAttribute('draggable'); // so it’s no longer draggable in the slot

  // 4) Append the clone into the slot
  e.currentTarget.appendChild(clonedFeed);

  console.debug(`[camera_feeds] Dropped feed "${feedId}" into slot #${e.currentTarget.id}.`);

  // Optional: remove the original feed from the preview if you want a true "move":
  // feedElement.remove();
}

/**
 * Automatically initialize camera feeds on DOMContentLoaded.
 */
document.addEventListener('DOMContentLoaded', initCameraFeeds);