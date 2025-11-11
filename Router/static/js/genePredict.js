function uploadCSV() {
  const fileInput = document.getElementById('csvFile');
  const token = localStorage.getItem("userToken");
  const file = fileInput.files[0];

  if (!file) {
    alert('請先選擇一個 CSV 檔案');
    return;
  }

  let formData = new FormData();
  formData.append("file", file);
  document.getElementById("result").innerHTML = "<p>分析中，請稍候. . .</p>";

  fetch(`/genePredict?ts=${Date.now()}`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${token}` },
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    const xValues = data.volcano_data.map(p => p.x);
    const yValues = data.volcano_data.map(p => p.y);

    const xMin = Math.min(...xValues) - 1;
    const xMax = Math.max(...xValues) + 1;
    const yMin = 0;
    const yMax = Math.max(...yValues) * 1.1; // 多留10%空間

    // 基因解釋對照表 (先寫死)
    const geneDescriptions = {
      "gene-FSTL5": "參與神經調控與訊號傳遞，可能透過下丘腦影響性腺激素的調節。",
      "gene-RPS20": "核糖體蛋白，屬於 40S 小亞基，與蛋白質合成相關；在高繁殖組上調可能反映神經細胞活性提升。",
      "novel433": "未知基因，於高繁殖組顯著上調，具潛在的神經調控角色，功能有待研究。",
      "gene-RPL24": "核糖體蛋白，屬於 60S 大亞基，參與蛋白質合成；表達上升可能反映下丘腦神經元代謝與活性增加。",
      "gene-GALNT13": "O-糖基轉移酶，負責蛋白質糖基化，可能影響神經肽或激素受體的穩定性與功能。",
      "gene-CTNND2": "細胞黏附與訊號傳遞相關基因，參與神經網絡連結，可能調控繁殖相關的神經迴路。",
      "gene-DTNB": "細胞骨架相關蛋白，可能維持神經細胞結構與功能穩定，支持下丘腦調控。",
      "gene-LHFPL6": "膜蛋白相關基因，功能尚未完全明確，可能參與神經突觸或訊號傳遞過程。",
      "gene-SLIT3": "神經導引分子，調控神經纖維連結與神經網絡形成，可能影響下丘腦的訊號傳遞。",
      "gene-RAPGEF2": "小GTP酶交換因子，參與神經訊號傳導與細胞極性調控，可能影響生殖相關神經路徑。",

      "novel1688": "未知基因，於低繁殖組顯著上調，可能具有抑制性調控功能。",
      "gene-SCRN1": "分泌調控相關基因，可能影響下丘腦神經激素或神經肽的釋放。",
      "gene-CADM2": "細胞黏附分子，參與神經元突觸連結，可能改變神經網絡穩定性與繁殖訊號。",
      "gene-ELMO1": "細胞運動與骨架重組調控因子，可能影響下丘腦神經細胞的功能或可塑性。",
      "gene-KCNK1": "雙孔鉀離子通道，調節神經細胞膜電位與興奮性，可能影響繁殖相關神經活性。",
      "novel710": "未知基因，於低繁殖組上調，功能待研究，可能參與抑制性調控。",
      "gene-EEF1A1": "蛋白質翻譯延伸因子，參與蛋白合成過程；於低繁殖組上調可能反映異常的代謝需求。",
      "novel908": "未知基因，具表現差異，可能參與神經調控或繁殖相關路徑。",
      "gene-GALNT17": "O-糖基轉移酶，參與神經肽或激素受體的修飾，可能影響繁殖訊號。",
      "gene-EXOC6B": "胞吐複合體成分，參與囊泡運輸與分泌，可能調控下丘腦神經激素的釋放途徑。"
    };
    // const geneDescriptions = {
    //   "gene-FSTL5": "細胞分化相關，參與細胞外基質調控",
    //   "gene-SCRN1": "囊泡分泌調控，影響細胞訊號傳遞",
    //   "gene-CADM2": "細胞黏附蛋白，與神經和能量代謝相關",
    //   "gene-ELMO1": "細胞吞噬與運動控制，與免疫有關",
    //   "gene-RPS20": "核糖體小次單元蛋白，參與蛋白質合成",
    //   "gene-RPL24": "核糖體大次單元蛋白，與蛋白質合成有關",
    //   "gene-KCNK1": "鉀離子通道，調控細胞膜電位",
    //   "novel1688": "新轉錄本，尚未有公開註解",
    //   "novel433": "新轉錄本，可能為物種特有基因",
    //   "novel710": "新轉錄本，潛在新標誌基因"
    // };

    // top_genes 表格
    let top_high_row = data.top_high_genes
      .map(item => `<tr><td>${item.gene}</td><td>${item.score}</td></tr>`)
      .join("");

    // top_genes 表格
    let top_low_row = data.top_low_genes
      .map(item => `<tr><td>${item.gene}</td><td>${item.score}</td></tr>`)
      .join("");

    // 高繁殖率 top10
    let highGenes = data.top_high_genes || [];
    let highTableHtml = `
      <h5>高繁殖率 Top 10 基因</h5>
      <table class="w3-table w3-striped w3-bordered w3-hoverable w3-card">
        <thead class="w3-light-grey">
          <tr>
            <th>Gene</th>
            <th>Score</th>
            <th>log2FC</th>
            <th>解釋</th>
          </tr>
        </thead>
        <tbody>
          ${highGenes.map((item, index) => `
            <tr class="${index % 2 === 0 ? 'w3-white' : 'w3-light-grey'}">
              <td>${item.gene}</td>
              <td>${item.score?.toFixed(4) ?? "NA"}</td>
              <td>${item.log2FC?.toFixed(3) ?? "NA"}</td>
              <td>${geneDescriptions[item.gene] || "暫無解釋"}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;

    // 低繁殖率 top10
    let lowGenes = data.top_low_genes || [];
    let lowTableHtml = `
      <h5>低繁殖率 Top 10 基因</h5>
      <table class="w3-table w3-striped w3-bordered w3-hoverable w3-card">
        <thead class="w3-light-grey">
          <tr>
            <th>Gene</th>
            <th>Score</th>
            <th>log2FC</th>
            <th>解釋</th>
          </tr>
        </thead>
        <tbody>
          ${lowGenes.map((item, index) => `
            <tr class="${index % 2 === 0 ? 'w3-white' : 'w3-light-grey'}">
              <td>${item.gene}</td>
              <td>${item.score?.toFixed(4) ?? "NA"}</td>
              <td>${item.log2FC?.toFixed(3) ?? "NA"}</td>
              <td>${geneDescriptions[item.gene] || "暫無解釋"}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;

    document.getElementById("result").innerHTML = `
      <h4>分析完成</h4>
      <p>檔案 : ${file.name}</p>
      <p>顯著基因數: ${data.sig_gene_count}</p>
      ${highTableHtml}
      ${lowTableHtml}
      <div id="volcanoChart" style="height: 500px; margin-top: 20px;"></div>
    `;
    // 確認有 volcano_data
    if (data.volcano_data && data.volcano_data.length > 0) {
       // 將資料分成顯著和非顯著兩組
      const significantGenes = data.volcano_data.filter(p => p.significant);
      const nonSignificantGenes = data.volcano_data.filter(p => !p.significant);

      Highcharts.chart('volcanoChart', {
        chart: {
          type: 'scatter',
          zoomType: 'xy'
        },
        title: {
          text: 'Volcano Plot'
        },
        xAxis: {
          title: { text: 'log2 Fold Change' },
          // plotLines: [
          //   { color: 'purple', dashStyle: 'Dash', width: 2, value: 1, label: {style:{color: 'purple', fontWeight: 'bold'} /*text: '2X up', align: 'left'*/ } },
          //   { color: 'blue', dashStyle: 'Dash', width: 2, value: -1, label: {style:{color: 'blue', fontWeight: 'bold'} /*text: '2X down', align: 'right'*/ } }
          // ],
          min: xMin,
          max: xMax
        },
        yAxis: {
          title: { text: '-log10(p-value)' },
          // plotLines: [
          //   // P=0.05 的線：**標籤放在圖表右側邊緣**
          //   { 
          //     color: 'green', 
          //     dashStyle: 'LongDashDot', 
          //     width: 2, 
          //     value: -Math.log10(0.05), 
          //   }
          // ],
          min: yMin,
          max: yMax
        },
        legend: {
          enabled: true,
          layout: 'vertical',
          align: 'right',
          verticalAlign: 'middle'
        },
        tooltip: {
          headerFormat: '',
          pointFormat: '<b>{point.gene}</b><br>log2FC: {point.x:.2f}<br>-log10(p): {point.y:.2f}'
        },
        plotOptions: {
          scatter: {
            marker: {
              radius: 4,
              symbol: 'circle',
              states: {
                hover: {
                  enabled: true,
                  lineColor: 'black'
                }
              }
            },
            states: {
              hover: {
                marker: { enabled: false }
              }
            }
          }
        },
        series: [
          {
            type: 'line',
            name: '標準線 (p = 0.05)', // 這會在圖例中顯示
            data: [
              // 第一個點不需要標籤
              { x: xMin, y: -Math.log10(0.05), marker: { enabled: false } }, // 使用 xMin 確保線從最左邊開始
              // 第二個點（最右邊）添加 dataLabels
              {
                x: xMax, y: -Math.log10(0.05), // 使用 xMax 確保線到最右邊結束
                dataLabels: {
                  enabled: true,
                  format: '標準線 (p = 0.05)',
                  align: 'right', // 讓文字靠右對齊
                  verticalAlign: 'middle', // 垂直置中
                  x: 5, // 調整文字與線的水平距離
                  style: { color: 'green', fontWeight: 'bold' }
                },
                marker: { enabled: false }
              }
            ],
            color: 'green',
            dashStyle: 'LongDashDot',
            lineWidth: 2,
            enableMouseTracking: false,
            marker: { enabled: false }, // 整條線的marker都禁用，避免兩個點顯示marker
            zIndex: 10
          },
          {
            type: 'line',
            name: '2X up',
            data: [[1, 0], [1, yMax]], // y 軸範圍要依照實際資料改
            color: 'purple',
            dashStyle: 'Dash',
            enableMouseTracking: false,
            marker: { enabled: false },
            zIndex: 10
          },
          {
            type: 'line',
            name: '2X down',
            data: [[-1, 0], [-1, yMax]], 
            color: 'blue',
            dashStyle: 'Dash',
            enableMouseTracking: false,
            marker: { enabled: false },
            zIndex: 10
          },
          {
            type: 'scatter',
            name: '顯著基因', // 紅點的說明
            data: significantGenes.map(p => ({
              x: p.x,
              y: p.y,
              gene: p.gene
            })),
            color: 'red', // 設定為紅色
            zIndex: 1
          },
          /*{
            type: 'scatter',
            name: '顯著基因',
            data: significantGenes.map(p => {
              const isTopHigh = highGenes.some(item => item.gene === p.gene);
              const isTopLow = lowGenes.some(item => item.gene === p.gene); // 雖然邏輯上顯著基因的log2FC>0和<0不會同時是Top High和Top Low，但保險起見可以都檢查

              let markerOptions = {};
              if (isTopHigh || isTopLow) {
                markerOptions = {
                  radius: 7, // 放大標記
                  symbol: 'diamond', // 改變形狀
                  lineColor: 'black',
                  lineWidth: 1
                };
              }

              return {
                x: p.x,
                y: p.y,
                gene: p.gene,
                color: (isTopHigh || isTopLow) ? 'orange' : 'red', // 給 Top 基因一個不同的顏色
                marker: markerOptions,
                dataLabels: { // 可以選擇顯示基因名稱
                  enabled: (isTopHigh || isTopLow),
                  format: '{point.gene}',
                  y: -10 // 調整位置
                }
              };
            }),
            color: 'red', // 這是預設的顯著基因顏色，會被上面的 `color` 覆蓋
            zIndex: 1
          },*/
          {
            type: 'scatter',
            name: '非顯著基因', // 灰點的說明
            data: nonSignificantGenes.map(p => ({
              x: p.x,
              y: p.y,
              gene: p.gene
            })),
            color: 'gray', // 設定為灰色
            zIndex: 1
          }
        ]
      });
    } else {
      document.getElementById("volcanoChart").innerHTML = "<p>沒有火山圖資料</p>";
    }

    fileInput.value = "";
  })

  .catch(error => {
    document.getElementById("result").innerHTML = "<p>上傳失敗，請重試。</p>";
    console.error("上傳錯誤:", error);
  });
}