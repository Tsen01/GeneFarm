let allData = [];
let filteredData = [];
let currentPage = 1;
const rowsPerPage = 10;
let collectionName = "";
let userRole = "";
let apiPrefix = "";

// 初始化：取得使用者角色與可用資料集
async function loadCollections() {
    try {
        userRole = localStorage.getItem("userRole") || "";

        if (!userRole) {
            const meRes = await fetch("/auth/me", {
                headers: { "Authorization": "Bearer " + localStorage.getItem("userToken") }
            });
            if (meRes.ok) {
                const meData = await meRes.json();
                userRole = meData.role || "";
                localStorage.setItem("userRole", userRole);
            }
        }

        apiPrefix = (userRole === "GeneticResearcher") ? "/gene" : "/farm";

        const res = await fetch(`${apiPrefix}/search_collections`, {
            headers: { "Authorization": "Bearer " + localStorage.getItem("userToken") }
        });
        if (!res.ok) throw new Error("載入 collections 失敗");
        const data = await res.json();

        const collections = data.collections || [];
        const select = document.getElementById("collectionSelect");
        select.innerHTML = "";

        const defaultOpt = document.createElement("option");
        defaultOpt.value = "";
        defaultOpt.textContent = "請選擇資料集";
        defaultOpt.selected = true;
        defaultOpt.disabled = true;
        select.appendChild(defaultOpt);

        collections.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col.name;
            opt.textContent = col.label;
            select.appendChild(opt);
        });

        select.addEventListener("change", async function () {
            collectionName = this.value;
            await loadFarmData(); // 抓取資料一次
            currentPage = 1;
            applyFilters(); // 初次篩選
        });

    } catch (e) {
        console.error("載入 collection 錯誤:", e);
    }
}

// 安全取得欄位
function getFieldsFromData(dataArray) {
    if (!Array.isArray(dataArray)) return [];
    for (const row of dataArray) {
        if (row && typeof row === "object") return Object.keys(row).filter(f => f !== "user_id");
    }
    return [];
}

// 載入資料
async function loadFarmData() {
    if (!collectionName) return;
    try {
        const response = await fetch(`${apiPrefix}/data/${collectionName}`, {
            headers: { "Authorization": "Bearer " + localStorage.getItem("userToken") }
        });
        if (!response.ok) throw new Error("載入資料失敗");
        const result = await response.json();

        allData = Array.isArray(result.data) ? result.data : [];
        filteredData = [...allData];

        if (allData.length === 0) return;

        const fields = result.fields || getFieldsFromData(allData);
        renderFieldFilters(fields);

    } catch (error) {
        console.error("載入資料錯誤:", error);
    }
}

// 動態欄位選單
function renderFieldFilters(fields) {
    const filterDiv = document.getElementById("fieldFilters");
    filterDiv.innerHTML = "";

    const allLabel = document.createElement("label");
    allLabel.className = "w3-margin-right";
    allLabel.innerHTML = `<input type="checkbox" id="checkAll" checked> 全部`;
    filterDiv.appendChild(allLabel);

    fields.forEach(field => {
        const label = document.createElement("label");
        label.className = "w3-margin-right";
        label.innerHTML = `<input type="checkbox" class="fieldCheckbox" value="${field}" checked> ${field}`;
        filterDiv.appendChild(label);
    });

    // 全選/取消全選
    document.getElementById("checkAll").addEventListener("change", function() {
        const checked = this.checked;
        document.querySelectorAll(".fieldCheckbox").forEach(cb => cb.checked = checked);
        applyFilters();
    });

    // 個別欄位勾選
    document.querySelectorAll(".fieldCheckbox").forEach(cb => {
        cb.addEventListener("change", function() {
            const allChecked = Array.from(document.querySelectorAll(".fieldCheckbox")).every(cb => cb.checked);
            document.getElementById("checkAll").checked = allChecked;
            applyFilters();
        });
    });
}

// 前端篩選函式（keyword + checkbox）
function applyFilters() {
    const keyword = document.getElementById("searchInput").value.trim().toLowerCase();
    const checkedFields = Array.from(document.querySelectorAll(".fieldCheckbox:checked")).map(cb => cb.value);

    if (checkedFields.length === 0) {
        filteredData = [...allData];
    } else {
        filteredData = allData.filter(row => {
            return checkedFields.some(f => {
                const val = row[f];
                return val != null && val.toString().toLowerCase().includes(keyword);
            });
        });
    }

    currentPage = 1;
    renderTablePage(currentPage);
    renderPaginationControls();
}

// 搜尋欄位輸入事件
document.getElementById("searchInput").addEventListener("input", applyFilters);

// 搜尋按鈕
document.getElementById("searchBtn").addEventListener("click", applyFilters);

// 分頁表格
function renderTablePage(page) {
    const tbody = document.getElementById("farmTableBody");
    tbody.innerHTML = "";

    if (!Array.isArray(filteredData) || filteredData.length === 0) return;

    const start = (page - 1) * rowsPerPage;
    const end = Math.min(start + rowsPerPage, filteredData.length);
    const fields = getFieldsFromData(filteredData);

    // 表頭
    const thead = document.getElementById("farmTableHead");
    thead.innerHTML = "";
    const headRow = document.createElement("tr");
    fields.forEach(field => {
        const th = document.createElement("th");
        th.textContent = field;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    // 表身
    filteredData.slice(start, end).forEach(rowData => {
        const row = document.createElement("tr");
        fields.forEach(field => {
            const cell = document.createElement("td");
            cell.textContent = rowData[field] ?? "—";
            row.appendChild(cell);
        });
        tbody.appendChild(row);
    });
}

// 分頁控制
function renderPaginationControls() {
    const paginationDiv = document.getElementById("paginationControls");
    paginationDiv.innerHTML = "";

    const totalPages = Math.ceil((filteredData?.length || 0) / rowsPerPage);
    if (totalPages === 0) return;

    const makeButton = (text, onClick, disabled = false) => {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.disabled = disabled;
        btn.className = "w3-button w3-border w3-round w3-small w3-margin";
        btn.onclick = onClick;
        return btn;
    };

    paginationDiv.appendChild(makeButton("⏮️ 第一頁", () => { currentPage = 1; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === 1));
    paginationDiv.appendChild(makeButton("◀️ 上一頁", () => { if(currentPage>1) currentPage--; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === 1));

    const select = document.createElement("select");
    select.className = "geneselect";
    for (let i = 1; i <= totalPages; i++) {
        const option = document.createElement("option");
        option.value = i;
        option.textContent = `第 ${i} 頁`;
        if (i === currentPage) option.selected = true;
        select.appendChild(option);
    }
    select.onchange = () => {
        currentPage = parseInt(select.value);
        renderTablePage(currentPage);
        renderPaginationControls();
    };
    paginationDiv.appendChild(select);

    paginationDiv.appendChild(makeButton("下一頁 ▶️", () => { if(currentPage<totalPages) currentPage++; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === totalPages));
    paginationDiv.appendChild(makeButton("最後一頁 ⏭️", () => { currentPage = totalPages; renderTablePage(currentPage); renderPaginationControls(); }, currentPage === totalPages));
}

// 初始化
window.addEventListener("DOMContentLoaded", loadCollections);