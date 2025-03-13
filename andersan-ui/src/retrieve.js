export function dateToJSTString(date) {

    // JSTの日時文字列に変換
    const jstString = date.toLocaleString('ja-JP', {
        timeZone: 'Asia/Tokyo',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false, // 24時間制
    });

    return jstString;
}

export function unixTimeToJSTString(unixTime) {
    // Unixタイムをミリ秒に変換
    const milliseconds = unixTime * 1000;

    // Dateオブジェクトを作成
    const date = new Date(milliseconds);

    return dateToJSTString(date);
}


function getOneHourAgo(date) {
    const newDate = new Date(date); // 元のDateオブジェクトを複製
    newDate.setHours(newDate.getHours() - 6);
    return newDate;
  }
  

export async function fetchData() {
    try {
        let oneHourAgo = getOneHourAgo(new Date());
        let isostring =  dateToJSTString(oneHourAgo).replace(/(\d+)\/(\d+)\/(\d+)\s(\d+):(\d+):(\d+)/, '$1-$2-$3T$4:$5:00+09:00');
        const response = await fetch(`http://172.23.78.65:8087/ox/v1a/kanagawa/${isostring}`); // APIエンドポイントを指定
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json(); // レスポンスをJSON形式で取得
        console.log(data); // 取得したデータをコンソールに表示
        return data; // 取得したデータを返す
    } catch (error) {
        console.error('データの取得中にエラーが発生しました:', error);
        // エラーハンドリングを行う
    }
  }
  
//   fetchData()->then(result=>{console.log(result)}); // 関数を実行

export async function fetchAddress(lon, lat){
    http://172.23.78.65:8087/loc/139.34944444444446/35.33555555555556
    try {
        const response = await fetch(`http://172.23.78.65:8087/loc/${lon}/${lat}`); // APIエンドポイントを指定
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json(); // レスポンスをJSON形式で取得
        console.log(data); // 取得したデータをコンソールに表示
        return data; // 取得したデータを返す
    } catch (error) {
        console.error('データの取得中にエラーが発生しました:', error);
        // エラーハンドリングを行う
    }
}