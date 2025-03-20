<script>
    import { onMount, afterUpdate } from 'svelte';
    import 'leaflet/dist/leaflet.css';
    import L from 'leaflet';
    import { fetchData, fetchAddress, fetchPtable } from './retrieve.js';
    import Chart from 'chart.js/auto';

    let map;
    let ox_dict;
    let address = "";
    let addr_dict = {};
    let ox_array;
    let p_array;
    let p_max = 0;
    let now = new Date("2015-07-27 06:00+09:00");
    let ptable;
    let myChart;
    let center = [35.331586, 139.349782]; // 地図の中心座標を保持する変数
    let debounceTimer; // デバウンスタイマー
    let updateFlag = false; // データ更新があったかどうかを示すフラグ

    function findMatchingRowIndex(array, targetValue1, targetValue2) {
        return array.findIndex(row => row[0] === targetValue1 && row[1] === targetValue2);
    }

    function formatTime(date) {
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    function formatStartTime(date) {
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        const hour = date.getHours().toString().padStart(2, '0');
        return `${month}月${day}日 ${hour}時時点`;
    }
    //グラフ描画関数を定義
    function drawChart(ox_array, p_array, now) {
        if (myChart) {
            myChart.destroy();
            myChart = null;
        }
        if (ox_array === undefined || p_array === undefined) return;
        let ticks = [];
        for (let hr = 1; hr <= 24; hr++) {
            ticks.push(hr);
        }
        let x = ticks.map((hr) => {
            const futureTime = new Date(now);
            futureTime.setHours(now.getHours() + hr);
            return `${formatTime(futureTime)} (+${hr})`;
        });

        let y1 = ox_array;
        let y2 = p_array;

        if (x.length > 0 && y1.length > 0 && y2.length > 0) {
            const ctx = document.getElementById('myChart').getContext('2d');
            myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: x,
                    datasets: [
                        {
                            label: 'OX Prediction (ppm)',
                            data: y1,
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1,
                            fill: false,
                            yAxisID: 'y',
                        },
                        {
                            label: 'Probability of exceeding 120ppm (%)',
                            data: y2,
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1,
                            fill: false,
                            yAxisID: 'y-right',
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Hours',
                            },
                        },
                        y: {
                            type: 'linear',
                            position: 'left',
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'OX (ppm)',
                            },
                        },
                        'y-right': {
                            type: 'linear',
                            position: 'right',
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Probability (%)',
                            },
                            min: 0,
                            max: 100,
                            ticks: {
                                stepSize: 20
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                    },
                },
                plugins: [{
                    beforeDraw: (chart) => {
                        const ctx = chart.canvas.getContext('2d');
                        ctx.save();
                        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                        ctx.fillRect(0, 0, chart.width, chart.height);
                        ctx.restore();
                    },
                }]
            });
        }
    }

    async function updateCenter() {
        updateFlag = false;
        const c = map.getCenter();
        center = [c.lat, c.lng];
        console.log("center updated")
        // 非同期処理をまとめて実行
        await Promise.all([
            fetchData(now).then(result => { ox_dict = result }),
            ptable === undefined ? fetchPtable().then(result => { ptable = result }) : Promise.resolve()
        ]);
        updateFlag = true;
    }

    async function updateAddress() {
        const c = map.getCenter();
        const latitude = c.lat;
        const longitude = c.lng;
         await fetchAddress(longitude, latitude).then(a => {
                address = a.address;
                addr_dict = a;
         });
    }

    function debounceUpdateCenter() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(updateCenter, 1000);
    }

    onMount(() => {
        // 平塚市中心部を初期表示、ズームレベルを12に設定
        map = L.map('map', { zoomControl: false }).setView([35.331586, 139.349782], 12);

        L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/blank/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution: '© <a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>'
        }).addTo(map);

        // moveend イベントのリスナーを追加
        map.on('moveend', updateAddress);
        // 初期表示時に updateCenter を実行
        updateCenter();
        updateAddress();
    });

    const moveToCurrentLocation = () => {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }

        const latitude = 35 + 20 / 60 + 8 / 3600;
        const longitude = 139 + 20 / 60 + 58 / 3600;
        map.setView([latitude, longitude], 12);
        updateCenter();
    };

    afterUpdate(() => {
        if (myChart) {
            myChart.resize();
        }
    });

    $: if (updateFlag) {
        now.setMinutes(0);
        now.setSeconds(0);
        now.setMilliseconds(0);
        //ox_arrayの計算
        if (ox_dict !== undefined) {
            if (addr_dict !== undefined) {
                let row = findMatchingRowIndex(ox_dict.data.XY, addr_dict.X, addr_dict.Y);
                ox_array = [];
                for (let hr = 1; hr <= 24; hr++) {
                    ox_array.push(Math.round(ox_dict.data[`+${hr}`][row]));
                }
            }
        }
        //p_arrayの計算
        if (ptable !== undefined && ox_array !== undefined) {
            p_array = [];
            for (let hr = 1; hr <= 24; hr++) {
                let ox = Math.floor(ox_array[hr - 1] / 5) * 5;
                let b = `(${ox}, ${hr})`;
                let a = "120";
                p_array.push(Math.round(ptable[a][b] * 100));
            }
            p_max = Math.max(...p_array);
        }
        //グラフ描画処理
        drawChart(ox_array, p_array, now);
    }
</script>

<div id="map">
    <div class="crosshair-container">
        <img class="crosshair" src="/images/crosshair.svg" alt="Target" />
    </div>
    <div class="address-overlay">{address}</div>
    <div class="pmax-overlay">
        <div class="pmax-label">本日中に注意報発令レベルに達する確率</div>
        <div class="pmax-value">{p_max}%</div>
        <div class="start-time-overlay">{formatStartTime(now)}</div>
    </div>
    <div class="tile-info-overlay">{addr_dict.X} {addr_dict.Y}</div>
    <div class="chart-container">
        <canvas id="myChart" class="chart-overlay"></canvas>
    </div>
    <button class="current-location-button" on:click={moveToCurrentLocation}>
        <img src="/images/near_me.svg" alt="現在地に移動" />
    </button>
</div>

<style>
    /* Resetting default margins and paddings */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    button {
        z-index: 10000;
    }

    #map {
        position: relative;
        height: 100vh;
        width: 100vw;
    }

    .crosshair-container {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 10000;
        pointer-events: none; /* アイコンがクリックイベントを邪魔しないようにする */
        display: flex; /* 画像を中央に配置するためのFlexbox */
        justify-content: center; /* 水平方向の中央揃え */
        align-items: center; /* 垂直方向の中央揃え */
    }

    .crosshair {
        width: 32px; /* アイコンのサイズを調整 */
        height: 32px;
    }

    .address-overlay {
        position: fixed;
        top: 10px;
        left: 10px;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 4px;
        border-radius: 4px;
        border: 1px solid black;
        font-size: 12px;
        z-index: 10000;
        white-space: nowrap;
        pointer-events: none; /* クリックイベントを邪魔しないようにする */
    }

    .chart-container {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50%;
        z-index: 900;
    }

    .chart-overlay {
        width: 100%;
        height: 100%;
        z-index: 900;
    }

    .pmax-overlay {
        position: absolute;
        top: 10%;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .pmax-label {
        font-size: 12pt;
        margin-bottom: 5px;
        color: black;
    }

    .pmax-value {
        font-size: 36pt;
        font-weight: bold;
        color: black;
    }

    .tile-info-overlay {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(255, 255, 255, 0.5);
        padding: 4px;
        border-radius: 4px;
        border: 1px solid black;
        font-size: 12px;
        z-index: 1000;
    }

    .start-time-overlay {
        font-size: 12pt;
        color: black;
    }

    .current-location-button {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background-color: #333;
        border: none;
        border-radius: 50%;
        padding: 6px;
        cursor: pointer;
        z-index: 10000;
        display: flex;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s ease-in-out;
    }

    .current-location-button img {
        width: 32px;
        height: 32px;
    }

    .current-location-button:hover {
        transform: scale(1.1);
    }
</style>
