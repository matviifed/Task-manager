// Store for tasks and current task being edited
let tasks = [];
let currentTaskId = null;

// Generate unique ID for tasks
function generateId() {
    return '_' + Math.random().toString(36).substr(2, 9);
}

// Add new task
function addTask() {
    const title = document.getElementById('taskInput').value.trim();
    const description = document.getElementById('taskDescription').value.trim();
    const priority = document.getElementById('taskPriority').value;
    const deadline = document.getElementById('taskDeadline').value;

    if (!title) {
        alert('Please enter a task title');
        return;
    }

    const task = {
        id: generateId(),
        title,
        description,
        priority,
        deadline,
        status: 'todo'
    };

    tasks.push(task);
    renderTask(task);
    clearInputs();
}

// Render single task
function renderTask(task) {
    const taskElement = document.createElement('li');
    taskElement.className = `task-item priority-${task.priority}-border`;
    taskElement.draggable = true;
    taskElement.id = task.id;
    taskElement.onclick = (e) => {
        // Prevent showing details when clicking action buttons
        if (!e.target.closest('.task-actions')) {
            showTaskDetails(task);
        }
    };
    taskElement.ondragstart = drag;

    const deadlineDate = task.deadline ? new Date(task.deadline).toLocaleDateString() : 'No deadline';

    taskElement.innerHTML = `
        <h3>${task.title}</h3>
        <div class="task-meta">
            <span class="priority-tag priority-${task.priority}">
                ${task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
            </span>
            <span class="deadline">
                <i class="far fa-clock"></i> ${deadlineDate}
            </span>
        </div>
        <div class="task-actions">
            <button class="action-btn complete-btn" onclick="completeTask('${task.id}')">
                <i class="fas fa-check"></i>
            </button>
            <button class="action-btn delete-btn" onclick="deleteTask('${task.id}')">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;

    const targetList = document.getElementById('todoList');
    targetList.appendChild(taskElement);
}

// Complete task
function completeTask(taskId) {
    const task = tasks.find(t => t.id === taskId);
    if (task) {
        task.completed = true;
        task.completedDate = new Date();
        
        // Remove from board
        const taskElement = document.getElementById(taskId);
        taskElement.remove();
        
        // Add to completed section
        renderCompletedTask(task);
        
        // Update tasks array
        tasks = tasks.filter(t => t.id !== taskId);
    }
}

// Render completed task
function renderCompletedTask(task) {
    const completedContainer = document.getElementById('completedTasksContainer');
    const taskElement = document.createElement('div');
    taskElement.className = 'completed-task-item';
    
    taskElement.innerHTML = `
        <h3 class="task-title">${task.title}</h3>
        <div class="task-meta">
            <span class="priority-tag priority-${task.priority}">
                ${task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
            </span>
        </div>
        <div class="completion-date">
            Completed on ${task.completedDate.toLocaleDateString()}
        </div>
        <button class="action-btn delete-btn" onclick="deleteCompletedTask(this)">
            <i class="fas fa-trash"></i>
        </button>
    `;
    
    completedContainer.appendChild(taskElement);
}

// Delete task (updated)
function deleteTask(taskId) {
    if (confirm('Are you sure you want to delete this task?')) {
        const taskElement = document.getElementById(taskId);
        if (taskElement) {
            taskElement.remove();
            tasks = tasks.filter(task => task.id !== taskId);
        }
        closePopup();
    }
}

// Delete completed task
function deleteCompletedTask(button) {
    if (confirm('Are you sure you want to delete this completed task?')) {
        const taskElement = button.closest('.completed-task-item');
        taskElement.remove();
    }
}

// Clear all completed tasks
function clearCompletedTasks() {
    if (confirm('Are you sure you want to clear all completed tasks?')) {
        const completedContainer = document.getElementById('completedTasksContainer');
        completedContainer.innerHTML = '';
    }
}

// Drag and Drop functions
function allowDrop(ev) {
    ev.preventDefault();
    const board = ev.target.closest('.board');
    if (board) {
        board.classList.add('drag-over');
    }
}

function drag(ev) {
    ev.dataTransfer.setData('text', ev.target.id);
    ev.target.classList.add('dragging');
}

function drop(ev) {
    ev.preventDefault();
    const board = ev.target.closest('.board');
    const taskList = board.querySelector('.task-list');
    
    const data = ev.dataTransfer.getData('text');
    const draggedElement = document.getElementById(data);
    
    draggedElement.classList.remove('dragging');
    board.classList.remove('drag-over');
    
    taskList.appendChild(draggedElement);
    
    // Update task status in data store
    const task = tasks.find(t => t.id === data);
    if (task) {
        task.status = board.querySelector('input').value.toLowerCase().replace(/\s+/g, '');
    }
}

// Close popup when clicking outside
window.onclick = function(event) {
    const popup = document.getElementById('taskPopup');
    if (event.target === popup) {
        closePopup();
    }
}