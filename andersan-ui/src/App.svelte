<script>
    import { onMount } from 'svelte';
    import 'leaflet/dist/leaflet.css';
    import L from 'leaflet';
    import { fetchData, fetchAddress, fetchPtable, unixTimeToJSTString } from './retrieve.js';
    import Chart from 'chart.js/auto';

    let map;
    let ox_dict;
    let address;
    let addr_dict;
    let ox_array;
    let p_array;
    let now;
    let X, Y;
    let ptable;
    // let longitude;
    // let latitude;

    function findMatchingRowIndex(array, targetValue1, targetValue2) {
        return array.findIndex(row => row[0] === targetValue1 && row[1] === targetValue2);
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
        // // Leafletのアイコンに関する設定（アイコンが表示されない問題の対策）
        // delete L.Icon.Default.prototype._getIconUrl;
        // L.Icon.Default.mergeOptions({
        //   iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
        //   iconUrl: require('leaflet/dist/images/marker-icon.png'),
        //   shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
        // });
    });
  
    // 現在地を取得して地図を移動する関数
    const moveToCurrentLocation = () => {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }
  
        const latitude = 35+20/60+8/3600;
        const longitude = 139+20/60+58/3600;
        // alert(longitude + " " + latitude)
        map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        fetchAddress(longitude, latitude).then(a=>{address=a.address; addr_dict = a;});
        fetchData().then(result=>{ox_dict=result});
        if ( ptable === undefined ){
            fetchPtable().then(result=>{ptable=result});
        }
        
        //   navigator.geolocation.getCurrentPosition(
        //     (position) => {
        //       const { latitude, longitude } = position.coords;
        //       map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        //       L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        //     },
        //     () => {
        //       alert('Unable to retrieve your location');
        //     }
        //   );
    };

    $:{
        // 
        // console.log(ox.data);
        if (ox_dict !== undefined){
            if (addr_dict !== undefined){
                X = addr_dict.X;
                Y = addr_dict.Y;
                // alert(X+":"+Y)
                let row = findMatchingRowIndex(ox_dict.data.XY, X, Y);
                ox_array = [];
                for (let hr = 1; hr <= 24; hr++) {
                    ox_array.push(Math.round(ox_dict.data[`+${hr}`][row]));
                }
                now = unixTimeToJSTString(ox_dict.spec.timestamp[0]);
            }
        }
        console.log([ptable, ox_array]);
        if ( ptable !== undefined ){
            if ( ox_array !== undefined ){
                p_array = [];
                let ticks = []
                for (let hr = 1; hr <= 24; hr++) {
                    let ox = Math.floor(ox_array[hr-1]/5)*5;
                    let b = `(${ox}, ${hr})`;
                    let a = "120";
                    p_array.push(Math.round(ptable[a][b]*100));
                    ticks.push(hr)
                }

                // JavaScript
                const ctx = document.getElementById('myChart').getContext('2d');
                const x = ticks
                const y = ox_array
                const z = p_array; // 実数値の系列

                let data = [];
                for (let i = 0; i < x.length - 1; i++) {
                    const gradient = ctx.createLinearGradient(x[i], y[i], x[i + 1], y[i + 1]);
                    gradient.addColorStop(0, `rgba(255, 0, 0, ${z[i]})`); // 開始色の透明度をzの値で調整
                    gradient.addColorStop(1, `rgba(0, 0, 255, ${z[i + 1]})`); // 終了色の透明度をzの値で調整
                    data.push({
                        x: [x[i], x[i + 1]],
                        y: [y[i], y[i + 1]],
                        borderColor: gradient,
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                    });
                }
                const myChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        datasets: data,
                    },
                    options: {
                        scales: {
                            x: { type: 'linear', position: 'bottom' },
                            y: { beginAtZero: true },
                        },
                        plugins: {
                            legend: { display: false },
                        },
                    },
                });
            }   


            
        }
    }

</script>
  
<div id="map" style="height: 67vh; width: 100vw;"></div>

<button on:click={moveToCurrentLocation}>現在地に移動</button><br />
現在地: {address}付近<br />
地理院タイル: {X} {Y} (Zoomレベル12)<br />
起点時刻: {now}<br />
OX予測: {ox_array} ppm<br />
120 ppm越え確率(%): {p_array}<br />
<canvas id="myChart"></canvas>


<style>
    button {
        z-index: 10;
    }
</style>