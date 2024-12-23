class TableManager {
    constructor() {
        this.customTableColumns = [];
        this.init();
    }

    init() {
        this.initializeEventListeners();
        this.initializeDragAndDrop();
    }

    initializeEventListeners() {
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => this.handleFileUpload(e));
        }

        const addColumnBtn = document.getElementById('add-column-btn');
        if (addColumnBtn) {
            addColumnBtn.addEventListener('click', () => this.addColumn());
        }

        const addRowBtn = document.getElementById('add-row-btn');
        if (addRowBtn) {
            addRowBtn.addEventListener('click', () => this.addRow());
        }

        const saveXlsBtn = document.getElementById('save-xls-btn');
        if (saveXlsBtn) {
            saveXlsBtn.addEventListener('click', () => this.saveAsExcel());
        }
    }

    initializeDragAndDrop() {
        const dropZone = document.querySelector('.file-upload-wrapper');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        dropZone.addEventListener('dragenter', () => {
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) {
                const fileInput = document.getElementById('file-upload');
                fileInput.files = e.dataTransfer.files;
                this.showFileName(file.name);
            }
        });

        // Handle file input change
        const fileInput = document.getElementById('file-upload');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.showFileName(file.name);
                }
            });
        }
    }

    showFileName(fileName) {
        const uploadIcon = document.querySelector('.upload-icon span');
        if (uploadIcon) {
            uploadIcon.textContent = `Selected file: ${fileName}`;
        }
    }

    async handleFileUpload(event) {
        event.preventDefault();
        const fileInput = document.getElementById('file-upload');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showNotification("Please select a file to upload.", "error");
            return;
        }

        const fileType = file.name.toLowerCase();
        try {
            if (fileType.endsWith('.csv')) {
                await this.handleCSVUpload(file);
            } else if (fileType.endsWith('.xlsx') || fileType.endsWith('.xls')) {
                await this.handleExcelUpload(file);
            } else {
                this.showNotification("Unsupported file type! Please upload a CSV or Excel file.", "error");
                return;
            }
            
            // Show success message
            this.showNotification("File uploaded successfully!", "success");
            
            // Reset the form
            setTimeout(() => {
                const uploadIcon = document.querySelector('.upload-icon span');
                if (uploadIcon) {
                    uploadIcon.textContent = "Drag and drop your file here or click to browse";
                }
                fileInput.value = '';
            }, 3000);

        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification('Error processing file', "error");
        }
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `upload-notification ${type}`;
        notification.textContent = message;
        
        // Remove existing notifications
        const existingNotification = document.querySelector('.upload-notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Add new notification
        const uploadArea = document.querySelector('.upload-area');
        if (uploadArea) {
            uploadArea.appendChild(notification);
            
            // Remove notification after 3 seconds
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
    }


    async handleCSVUpload(file) {
        const content = await this.readFileAsText(file);
        const rows = content.split('\n')
            .map(row => row.split(',').map(cell => cell.trim()))
            .filter(row => row.length > 0 && row.some(cell => cell !== ''));
        this.showFilePreview(rows);
    }

    async handleExcelUpload(file) {
        const data = await this.readFileAsBinary(file);
        const workbook = XLSX.read(data, { type: 'binary' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const rows = XLSX.utils.sheet_to_json(worksheet, { header: 1 })
            .filter(row => row.length > 0 && row.some(cell => cell !== undefined));
        this.showFilePreview(rows);
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }

    readFileAsBinary(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsBinaryString(file);
        });
    }

    showFilePreview(rows) {
        const previewContainer = document.getElementById('file-preview');
        if (!previewContainer) return;

        previewContainer.innerHTML = '';
        const table = document.createElement('table');
        table.className = 'preview-table';
        
        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        rows[0].forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        rows.slice(1).forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell || '';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        previewContainer.appendChild(table);
    }

    addColumn() {
        const columnNameInput = document.getElementById('column-name');
        const columnName = columnNameInput.value.trim();
        
        if (!columnName) return;

        this.customTableColumns.push(columnName);
        
        // Update table header
        const tableColumns = document.getElementById('table-columns');
        if (tableColumns) {
            const th = document.createElement('th');
            th.textContent = columnName;
            tableColumns.appendChild(th);

            // Add new cell to each existing row
            const tbody = document.querySelector('#custom-table tbody');
            if (tbody) {
                tbody.querySelectorAll('tr').forEach(row => {
                    const td = document.createElement('td');
                    td.innerHTML = '<input type="text" placeholder="Enter data">';
                    row.appendChild(td);
                });
            }
        }

        columnNameInput.value = '';
    }

    addRow() {
        const tbody = document.querySelector('#custom-table tbody');
        if (!tbody) return;

        const tr = document.createElement('tr');
        this.customTableColumns.forEach(() => {
            const td = document.createElement('td');
            td.innerHTML = '<input type="text" placeholder="Enter data">';
            tr.appendChild(td);
        });
        
        tbody.appendChild(tr);
    }

    async saveAsExcel() {
        try {
            // Get table data
            const headers = Array.from(document.querySelectorAll('#table-columns th'))
                .map(th => th.textContent);
            
            const rows = Array.from(document.querySelectorAll('#custom-table tbody tr'))
                .map(row => Array.from(row.querySelectorAll('input'))
                    .map(input => input.value));

            const data = [headers, ...rows];

            // Create and save Excel file
            const ws = XLSX.utils.aoa_to_sheet(data);
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
            XLSX.writeFile(wb, 'custom_table.xlsx');
            
            
            this.showAlert('Table saved successfully as Excel file!', 'success');
        } catch (error) {
            console.error('Error saving custom table:', error);
            this.showAlert('Error saving table as Excel file');
        }
    }

    showAlert(message, type = 'error') {
        alert(message);
    }
}

// Initialize the table manager
document.addEventListener('DOMContentLoaded', () => {
    window.tableManager = new TableManager();
});