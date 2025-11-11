// 切換匯入 / 手動
function setActiveButton(activeId) {
  document.getElementById("btnImport").style.backgroundColor = "#ccc";
  document.getElementById("btnManual").style.backgroundColor = "#ccc";
  document.getElementById(activeId).style.backgroundColor = "#0dac9c";
}

function showImport() {
  document.getElementById("importSection").style.display = "flex";
  document.getElementById("manualSection").style.display = "none";
  setActiveButton("btnImport");
}

function showManual() {
  document.getElementById("importSection").style.display = "none";
  document.getElementById("manualSection").style.display = "block";
  setActiveButton("btnManual");
}

// 預設顯示「匯入一個資料集」
document.addEventListener("DOMContentLoaded", () => {
  showImport();
});


// 手動新增欄位區
function toggleManualMode() {
  const mode = document.getElementById("manualMode").value;
  document.getElementById("newCollectionDiv").style.display = (mode === "new") ? "block" : "none";
  document.getElementById("existingCollectionDiv").style.display = (mode === "existing") ? "block" : "none";

  const container = document.getElementById("fieldContainer");
  container.innerHTML = "<h5>欄位資料</h5>";

  // 如果使用現有資料集，載入欄位
  if (mode === "existing") {
    const select = document.getElementById("existingCollectionSelect");
    if (select.value) {
      const event = new Event('change');
      select.dispatchEvent(event);
    }
  }
}

// 新增一組欄位 (for 新資料集)
function addField() {
  const container = document.getElementById("fieldContainer");
  const row = document.createElement("div");
  row.className = "w3-row-padding";
  row.innerHTML = `
    <div class="w3-half">
      <input class="w3-input w3-border" type="text" placeholder="欄位名稱" name="fieldName">
    </div>
    <div class="w3-half">
      <input class="w3-input w3-border" type="text" placeholder="欄位值" name="fieldValue">
    </div>
  `;
  container.appendChild(row);
}

// 手動新增表單提交
document.getElementById("manualForm").addEventListener("submit", async function(e) {
  e.preventDefault();
  const token = localStorage.getItem("userToken");
  if (!token) return alert("請先登入！");

  const mode = document.getElementById("manualMode").value;
  let collectionName = (mode === "new") ? document.getElementById("newCollectionName").value.trim()
                                        : document.getElementById("existingCollectionSelect").value;
  if (!collectionName) return alert("請輸入資料集名稱或選擇現有資料集");

  // 收集欄位資料
  const fields = {};
  document.querySelectorAll('#fieldContainer .w3-row-padding').forEach(row => {
    const inputs = row.querySelectorAll("input");
    if (inputs.length === 2) {
      const fieldName = inputs[0].value.trim();
      const fieldValue = inputs[1].value.trim();
      if (fieldName) fields[fieldName] = fieldValue;
    }
  });

  try {
    const res = await fetch(`/farm/manual_add/${collectionName}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify(fields)
    });

    const result = await res.json();
    if (res.ok) {
      alert("新增成功！");

      // 新增成功後，清空欄位值
      document.querySelectorAll('#fieldContainer input[name]').forEach(input => input.value = "");

      // 更新 collection 清單並顯示最新資料
      await refreshCollections(collectionName);
    } else {
      alert("新增失敗：" + (result.detail || JSON.stringify(result)));
    }
  } catch (err) {
    console.error("送出失敗:", err);
    alert("送出失敗，請查看 console");
  }
});

// 匯入資料集 (上傳檔案)
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const token = localStorage.getItem("userToken");
  if (!token) return alert("請先登入！");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/farm/upload", {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}` },
      body: formData
    });

    const result = await res.json();
    if (res.ok) {
      alert("匯入成功！");
      await refreshCollections(result.collection);
    } else {
      alert("匯入失敗：" + (result.detail || JSON.stringify(result)));
    }
  } catch (err) {
    console.error("匯入失敗:", err);
    alert("匯入失敗，請查看 console");
  }
}


//  全域狀態 
let allData = [];
let currentFields = [];
let currentPage = 1;
let currentCollection = "";
const rowsPerPage = 10;


// 載入某個 collection 的資料
async function loadCollectionData(collectionName) {
  const token = localStorage.getItem("userToken");
  if (!token) return;

  try {
    const res = await fetch(`/farm/data/${collectionName}`, {
      headers: { "Authorization": `Bearer ${token}` }
    });

    const result = await res.json();
    if (res.ok || res.status === 200) {
      // 後端回傳 {fields, data}
      allData = result.data || [];
      currentFields = result.fields || [];
      currentCollection = collectionName;
      currentPage = 1;
      renderTablePage(currentPage);
    } else {
      console.error("載入資料失敗：", result);
      allData = [];
      currentFields = [];
      renderTablePage(1);
    }
  } catch (err) {
    console.error("無法載入資料：", err);
    allData = [];
    currentFields = [];
    renderTablePage(1);
  }
}

// 載入 collection 清單
async function refreshCollections(selectedCollection = null) {
  const token = localStorage.getItem("userToken");
  if (!token) return;

  const select = document.getElementById("collectionSelect");
  const existingSelect = document.getElementById("existingCollectionSelect");

  try {
    const res = await fetch("/farm/collections", {
      headers: { "Authorization": `Bearer ${token}` },
    "Content-Type": "application/json"
    });
    const collectionsData = await res.json();
    const collections = Array.isArray(collectionsData) ? collectionsData : [];

    select.innerHTML = "";
    existingSelect.innerHTML = "";

    collections.forEach(name => {
      const opt1 = document.createElement("option");
      opt1.value = name;
      opt1.textContent = name;
      select.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = name;
      opt2.textContent = name;
      existingSelect.appendChild(opt2);
    });

    const target = selectedCollection || collections[0];
    if (target) {
      select.value = target;
      loadCollectionData(target);
    }

    select.onchange = () => loadCollectionData(select.value);

    existingSelect.onchange = async function() {
      const collectionName = this.value;
      if (!collectionName) return;

      try {
        const res = await fetch(`/farm/fields/${collectionName}`, {
          headers: { "Authorization": `Bearer ${token}` }
        });
        const fields = await res.json();

        const container = document.getElementById("fieldContainer");
        container.innerHTML = "<h5>欄位資料</h5>";

        fields.forEach(field => {
          const row = document.createElement("div");
          row.className = "w3-row-padding";
          row.innerHTML = `
            <div class="w3-half">
              <input class="w3-input w3-border" type="text" value="${field}" readonly>
            </div>
            <div class="w3-half">
              <input class="w3-input w3-border" type="text" placeholder="請輸入 ${field}" name="${field}">
            </div>
          `;
          container.appendChild(row);
        });

      } catch (err) {
        console.error("載入欄位失敗:", err);
      }
    };

  } catch (err) {
    console.error("無法載入資料集清單：", err);
  }
}


// 編輯資料集
function editCollection() {
  const select = document.getElementById("collectionSelect");
  const collectionName = select.value;
  if (!collectionName) {
    alert("請先選擇一個資料集");
    return;
  }

  // 跳轉到 EditData.html，並帶上 collection 參數
  window.location.href = `/AnimalManager/Edit/${collectionName}`;
}

// 刪除資料集
async function deleteCollection() {
  const select = document.getElementById("collectionSelect");
  const collectionName = select.value;

  if (!collectionName) {
    alert("請先選擇資料集");
    return;
  }

  if (!confirm(`確定要刪除資料集「${collectionName}」嗎？此操作無法復原！`)) return;

  try {
    const token = localStorage.getItem("userToken"); // JWT
    const res = await fetch(`/farm/delete_collection/${collectionName}`, {
      method: "DELETE",
      headers: { "Authorization": "Bearer " + token }
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || "刪除失敗");
    }

    alert(`資料集「${collectionName}」已刪除`);
    location.reload();  // 重新整理頁面

  } catch (err) {
    console.error(err);
    alert("刪除資料集失敗：" + err.message);
  }
}


//  分頁表格 
function renderTablePage(page) {
  const tbody = document.getElementById("tableBody");
  tbody.innerHTML = "";

  if (!allData || allData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="99" class="w3-center">尚未匯入羊隻資料</td></tr>`;
    document.getElementById("paginationControls").innerHTML = "";
    return;
  }

  const start = (page - 1) * rowsPerPage;
  const end = Math.min(start + rowsPerPage, allData.length);

  // 取得欄位
  const keys = currentFields;

  // 表頭
  const header = document.getElementById("tableHeader");
  header.innerHTML = "";
  const tr = document.createElement("tr");
  keys.forEach(k => {
    const th = document.createElement("th");
    th.textContent = k;
    tr.appendChild(th);
  });
  header.appendChild(tr);

  // 表格內容
  for (let i = start; i < end; i++) {
    const row = document.createElement("tr");
    keys.forEach(key => {
      const td = document.createElement("td");
      td.textContent = allData[i][key] ?? "";
      row.appendChild(td);
    });
    tbody.appendChild(row);
  }

  renderPaginationControls();
}

function renderPaginationControls() {
  const div = document.getElementById("paginationControls");
  div.innerHTML = "";

  const totalPages = Math.ceil(allData.length / rowsPerPage);
  if (totalPages <= 1) return;

  const makeBtn = (text, fn, disabled) => {
    const btn = document.createElement("button");
    btn.textContent = text;
    btn.disabled = disabled;
    btn.className = "w3-button w3-border w3-small w3-margin";
    btn.onclick = fn;
    return btn;
  };

  div.appendChild(makeBtn("⏮️ 第一頁", () => { currentPage=1; renderTablePage(currentPage); }, currentPage===1));
  div.appendChild(makeBtn("◀️ 上一頁", () => { currentPage--; renderTablePage(currentPage); }, currentPage===1));

  const select = document.createElement("select");
  for (let i=1;i<=totalPages;i++){
    const option = document.createElement("option");
    option.value=i;
    option.textContent=`第 ${i} 頁`;
    if(i===currentPage) option.selected=true;
    select.appendChild(option);
  }
  select.onchange = () => {
    currentPage = Number(select.value);
    renderTablePage(currentPage);
  };
  div.appendChild(select);

  div.appendChild(makeBtn("下一頁 ▶️", () => { currentPage++; renderTablePage(currentPage); }, currentPage===totalPages));
  div.appendChild(makeBtn("最後頁 ⏭️", () => { currentPage=totalPages; renderTablePage(currentPage); }, currentPage===totalPages));
}

// 初始化 
document.addEventListener("DOMContentLoaded", () => {
  showImport();
  refreshCollections();
  document.getElementById("manualMode").addEventListener("change", toggleManualMode);
});