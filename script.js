
let tasks = [];
let currentTaskId = null;


function generateId() {
    return '_' + Math.random().toString(36).substr(2, 9);
}

//Tasks scritps for creating, completing and deleting

async function addTask() {
    const title = document.getElementById("taskInput").value.trim();
    const description = document.getElementById("taskDescription").value.trim();
    const priority = document.getElementById("taskPriority").value;
    const deadline = document.getElementById("taskDeadline").value;

    if (!title) {
        alert("Please enter a task title");
        return;
    }

    const taskData = { title, description, priority, deadline };

    try {
        const response = await fetch("http://localhost:5000/tasks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(taskData),
        });

        if (!response.ok) {
            throw new Error('Failed to add task');
        }

        const newTask = await response.json();
        tasks.push(newTask);
        renderTask(newTask);
        clearInputs();
    } catch (error) {
        console.error('Error adding task:', error);
        alert('Failed to add task. Please try again.');
    }
}


let currentTask = null;

function showTaskDetails(task) {
    const popup = document.getElementById('taskPopup');
    const title = document.getElementById('popupTitle');
    const description = document.getElementById('popupDescription');
    const tipsContainer = document.getElementById('tipsContainer');
    const tipsContent = document.getElementById('tipsContent');

    currentTask = task;
    currentTaskId = task._id;
    title.textContent = task.title;
    description.textContent = task.description || 'No description provided';
    
    tipsContainer.style.display = 'none';
    tipsContent.textContent = '';

    popup.style.display = 'block';
}

function closePopup() {
    const popup = document.getElementById('taskPopup');
    const tipsContainer = document.getElementById('tipsContainer');
    const tipsContent = document.getElementById('tipsContent');
    
    popup.style.display = 'none';
    tipsContainer.style.display = 'none';
    tipsContent.textContent = '';
    currentTask = null;
    currentTaskId = null;
}

function renderTask(task) {
    const taskElement = document.createElement('li');
    taskElement.className = `task-item priority-${task.priority}-border`;
    taskElement.draggable = true;
    taskElement.id = task._id; 
    taskElement.onclick = (e) => {
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
            <button class="action-btn complete-btn" onclick="completeTask('${task._id}')">
                <i class="fas fa-check"></i>
            </button>
            <button class="action-btn delete-btn" onclick="deleteTask('${task._id}')">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;

    const targetList = document.getElementById('todoList');
    targetList.appendChild(taskElement);
}

async function completeTask(taskId) {
    const task = tasks.find(t => t._id === taskId);
    if (!task) return;

    const completedTask = {
        id: task._id, 
        title: task.title,
        description: task.description,
        priority: task.priority,
        deadline: task.deadline
    };

    await fetch("http://localhost:5000/completed-tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(completedTask),
    });


    document.getElementById(taskId).remove();
    tasks = tasks.filter(t => t._id !== taskId);

    fetchCompletedTasks();
}

async function deleteTask(taskId) {
    if (!taskId && currentTaskId) {
        taskId = currentTaskId;
    }
    
    if (confirm("Are you sure you want to delete this task?")) {
        try {
            const response = await fetch(`http://localhost:5000/tasks/${taskId}`, { 
                method: "DELETE" 
            });
            
            if (response.ok) {
                document.getElementById(taskId)?.remove();
                tasks = tasks.filter(task => task._id !== taskId);
                closePopup();
            } else {
                console.error('Failed to delete task');
            }
        } catch (error) {
            console.error('Error deleting task:', error);
        }
    }
}

async function deleteCompletedTask(taskId) {
    if (confirm("Are you sure you want to delete this completed task?")) {
        await fetch(`http://localhost:5000/completed-tasks/${taskId}`, { method: "DELETE" });
        fetchCompletedTasks(); 
    }
}

async function clearCompletedTasks() {
    if (confirm("Are you sure you want to clear all completed tasks?")) {
        await fetch("http://localhost:5000/completed-tasks", { method: "DELETE" });
        fetchCompletedTasks(); 
    }
}

async function fetchTasks() {
    const response = await fetch("http://localhost:5000/tasks");
    const data = await response.json();

    tasks = data; 
    const todoList = document.getElementById("todoList");
    todoList.innerHTML = ""; 

    tasks.forEach(task => renderTask(task));
}




document.addEventListener("DOMContentLoaded", fetchTasks);

async function fetchCompletedTasks() {
    const response = await fetch("http://localhost:5000/completed-tasks");
    const completedTasks = await response.json();
    
    const completedContainer = document.getElementById("completedTasksContainer");
    completedContainer.innerHTML = ""; 

    completedTasks.forEach(task => renderCompletedTask(task));
}

document.addEventListener("DOMContentLoaded", fetchCompletedTasks);

function renderCompletedTask(task) {
    const completedContainer = document.getElementById("completedTasksContainer");
    const taskElement = document.createElement("div");
    taskElement.className = "completed-task-item";
    
    taskElement.innerHTML = `
        <h3 class="task-title">${task.title}</h3>
        <div class="task-meta">
            <span class="priority-tag priority-${task.priority}">
                ${task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
            </span>
        </div>
        <div class="completion-date">
            Completed on ${new Date(task.completedDate).toLocaleDateString()}
        </div>
        <button class="action-btn delete-btn" onclick="deleteCompletedTask('${task._id}')">
            <i class="fas fa-trash"></i>
        </button>
    `;

    completedContainer.appendChild(taskElement);
}


async function getTaskTips() {
    const tipsContainer = document.getElementById('tipsContainer');
    const tipsContent = document.getElementById('tipsContent');
    
    if (!currentTask) return;

    try {
        tipsContainer.style.display = 'block';
        tipsContent.innerHTML = '<div class="loading">Getting tips...</div>';

        const response = await fetch('http://localhost:5000/tasks/tips', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: currentTask.title,
                description: currentTask.description
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to get tips');
        }

        const data = await response.json();
        
        tipsContainer.style.display = 'block';
        tipsContent.textContent = data.tips;
    } catch (error) {
        console.error('Error getting tips:', error);
        tipsContent.textContent = 'Failed to get tips. Please try again.';
    }
}

//Scripts for adding and deleting boards

function addBoard() {
    const boardContainer = document.getElementById('boardContainer');
    const boardId = generateId(); 
    const board = document.createElement('div');
    board.className = 'board';
    board.setAttribute('ondrop', 'drop(event)');
    board.setAttribute('ondragover', 'allowDrop(event)');

    board.innerHTML = `
        <div class="board-header">
            <input type="text" value="New Board" oninput="renameBoard(this)">
            <button class="delete-btn" onclick="deleteBoard(this)">
                <i class="fa fa-trash"></i>
            </button>
        </div>
        <ul class="task-list"></ul>
    `;

    boardContainer.appendChild(board);
}


function deleteBoard(button) {
    const board = button.closest('.board');
    if (board && confirm('Are you sure you want to delete this board?')) {
        board.remove();
    }
}


//Scripts for drag and drop

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
    

    const task = tasks.find(t => t.id === data);
    if (task) {
        task.status = board.querySelector('input').value.toLowerCase().replace(/\s+/g, '');
    }
}


window.onclick = function(event) {
    const popup = document.getElementById('taskPopup');
    if (event.target === popup) {
        closePopup();
    }
}

//Additional script to clear inputs

function clearInputs() {
    document.getElementById("taskInput").value = "";
    document.getElementById("taskDescription").value = "";
    document.getElementById("taskPriority").value = "low";
    document.getElementById("taskDeadline").value = "";
}