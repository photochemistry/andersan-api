<script>
    import { onMount, afterUpdate } from 'svelte';
    import 'leaflet/dist/leaflet.css';
    import L from 'leaflet';
    import { fetchData, fetchAddress, fetchPtable } from './retrieve.js';
    import Chart from 'chart.js/auto';

    let map;
    let ox_dict;
    let address = ""; // initialize address to empty string
    let addr_dict;
    let ox_array;
    let p_array; // Renamed to p_array from z for consistency
    let p_max;
    let now;
    let X, Y;
    let ptable;
    let myChart; // Add myChart variable

    function findMatchingRowIndex(array, targetValue1, targetValue2) {
        return array.findIndex(row => row[0] === targetValue1 && row[1] === targetValue2);
    }

    function formatTime(date) {
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    onMount(() => {
        // 地図の初期化
        map = L.map('map').setView([0, 0], 13); // 初期位置は適宜設定

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        delete L.Icon.Default.prototype._getIconUrl;
        L.Icon.Default.mergeOptions({
            iconRetinaUrl: '/images/marker-icon-2x.png',
            iconUrl: '/images/marker-icon.png',
            shadowUrl: '/images/marker-shadow.png',
        });
    });

    // 現在地を取得して地図を移動する関数
    const moveToCurrentLocation = () => {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }

        const latitude = 35 + 20 / 60 + 8 / 3600;
        const longitude = 139 + 20 / 60 + 58 / 3600;
        map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        fetchAddress(longitude, latitude).then(a => { address = a.address; addr_dict = a; });
        let now_aux = new Date();
        fetchData(now_aux).then(result => { ox_dict = result });
        if (ptable === undefined) {
            fetchPtable().then(result => { ptable = result });
        }
    };

    afterUpdate(() => {
        // Resize the chart after the chart is created and after svelte's DOM has been updated
        if (myChart) {
            myChart.resize();
        }
    });

    $: {
        if (myChart) {
            myChart.destroy();
        }

        if (ox_dict !== undefined) {
            if (addr_dict !== undefined) {
                X = addr_dict.X;
                Y = addr_dict.Y;
                let row = findMatchingRowIndex(ox_dict.data.XY, X, Y);
                ox_array = [];
                for (let hr = 1; hr <= 24; hr++) {
                    ox_array.push(Math.round(ox_dict.data[`+${hr}`][row]));
                }
                now = new Date(ox_dict.spec.timestamp[0] * 1000);
                now.setMinutes(0);
                now.setSeconds(0);
                now.setMilliseconds(0);
            }
        }

        if (ptable !== undefined) {
            if (ox_array !== undefined) {
                p_array = []; // Initialize p_array here
                let ticks = [];
                for (let hr = 1; hr <= 24; hr++) {
                    let ox = Math.floor(ox_array[hr - 1] / 5) * 5;
                    let b = `(${ox}, ${hr})`;
                    let a = "120";
                    p_array.push(Math.round(ptable[a][b] * 100));
                    ticks.push(hr);
                }
                p_max = Math.max(...p_array)

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
                            responsive: true, // Enable responsiveness.
                            maintainAspectRatio: false, // Disable aspect ratio to make the chart as high as the container
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
                                ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; // semi-transparent white
                                ctx.fillRect(0, 0, chart.width, chart.height);
                                ctx.restore();
                            },
                        }]
                    });
                }
            }
        }
    }
</script>

<div id="map" style="height: 67vh; width: 100vw;">
    <div class="address-overlay">{address}</div>
    <div class="pmax-overlay">
        <div class="pmax-label">本日中に注意報が発令される確率</div>
        <div class="pmax-value">{p_max}%</div>
    </div>
    <div class="chart-container">
        <canvas id="myChart" class="chart-overlay"></canvas>
    </div>
</div>

<button on:click={moveToCurrentLocation}>現在地に移動</button><br />
地理院タイル: {X} {Y} (Zoomレベル12)<br />
起点時刻: {now}<br />

<style>
    button {
        z-index: 10;
    }

    #map {
        position: relative; /* Make the map a positioning context */
    }

    .address-overlay {
        position: absolute; /* Absolute positioning within the map */
        top: 50%; /* Center vertically */
        left: 50%; /* Center horizontally */
        transform: translate(-50%, -50%); /* Adjust for element's size */
        background-color: rgba(255, 255, 255, 0.5); /* Semi-transparent white */
        padding: 4px;
        border-radius: 4px;
        border: 1px solid black;
        font-size: 12px;
        text-align: center; /* Center text */
        z-index: 1000; /* Ensure it's on top */
    }

    .chart-container {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 40%;
        z-index: 900; /* Make sure it's above the map tiles but below the address */
    }

    .chart-overlay {
        width: 100%; /* Make it as wide as the map */
        height: 100%; /* Make it 40% of the map's height */
        z-index: 900; /* Make sure it's above the map tiles but below the address */
    }
    .pmax-overlay {
        position: absolute;
        top: 25%;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        z-index: 1000;
    }

    .pmax-label {
        font-size: 12pt;
        margin-bottom: 5px; /* Add a little space between the label and the value */
        color: black;
    }

    .pmax-value {
        font-size: 36pt;
        font-weight: bold;
        color: black;
    }
</style>
