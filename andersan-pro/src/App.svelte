<script>
	import axios from 'axios';
	import { onMount } from 'svelte';

	let prefecture = 'kanagawa'; // デフォルトの都道府県
	let model = 'v0a'; // モデルを v0a に固定
	let oxData = {}; // 初期値を空のオブジェクトに変更
	let loading = false;
	let error = null;
	let selectedDateTime = 'now'; // デフォルトは現在時刻
	let ptable = null; // 確率表を保持する状態変数
	let ptableError = null; // 確率表取得時のエラーを保持する状態変数
	let ptableLoading = false; // 確率表取得中のローディング状態
	let ptableRows = []; // 確率表の行データを保持する変数
	let ptableColumns = []; // 確率表のカラムデータを保持する変数
	const zValue = 120; // z の値を 120 に固定

	// 日時を ISO 8601 形式に変換する関数 (JST に対応)
	function getISO8601(date) {
		const jstOffset = 9 * 60; // JST は UTC+9 なので、9時間分のオフセットを分単位で設定
		const utc = date.getTime() + (date.getTimezoneOffset() * 60000); // UTC に変換
		const jst = new Date(utc + (jstOffset * 60000)); // JST に変換

		const year = jst.getFullYear();
		const month = (jst.getMonth() + 1).toString().padStart(2, '0'); // 月は 0-11 なので +1 する
		const day = jst.getDate().toString().padStart(2, '0');
		const hour = jst.getHours().toString().padStart(2, '0');
		const minute = jst.getMinutes().toString().padStart(2, '0');

		return `${year}-${month}-${day}T${hour}:${minute}:00+09:00`;
	}

	// データを取得する関数
	async function fetchData() {
		loading = true;
		error = null;
		if (Object.keys(oxData).length === 0) {
			oxData = {}; // 取得前に空オブジェクトに初期化
		}

		try {
			let formattedDatehour;
			if (selectedDateTime === 'now') {
				formattedDatehour = getISO8601(new Date()); // 現在時刻を取得 (JST)
			} else {
				formattedDatehour = selectedDateTime;
			}
			const response = await axios.get(
				// `http://172.23.78.71:8087/ox/${model}/${prefecture}/${formattedDatehour}`
				`http://192.168.3.234:8087/ox/${model}/${prefecture}/${formattedDatehour}`
			);

			// データの整形
			const all_data = response.data;
			const data = all_data.data;

			oxData = data; // data をそのまま oxData に代入

		} catch (err) {
			if (err.response) {
				// サーバーがエラーレスポンスを返した場合
				error = `データ取得エラー: ${err.response.status} - ${err.response.statusText}`;
			} else if (err.request) {
				// リクエストが送信されたが、レスポンスが受信されなかった場合
				error = 'データ取得エラー: サーバーからの応答がありません。';
			} else {
				// エラーをセットアップする際に何か問題が発生した場合
				error = 'データ取得エラー: リクエストを送信できませんでした。';
			}
		} finally {
			loading = false;
		}
	}

	// 確率表を取得する関数
	async function fetchPtable() {
		ptableLoading = true;
		ptableError = null;
		try {
			const response = await axios.get(
				// `http://172.23.78.71:8087/ptable/${model}`
				`http://192.168.3.234:8087/ptable/${model}`
			);
			// JSON データを解析して、行と列のデータに変換
			const jsonData = response.data;
			ptableColumns = Object.keys(jsonData);
			const indexKeys = new Set();
			ptableColumns.forEach(col => {
				Object.keys(jsonData[col]).forEach(key => indexKeys.add(key));
			});
			ptableRows = Array.from(indexKeys).map(index => {
				const [index1, index2] = index.slice(1, -1).split(',').map(Number); // 修正箇所
				const row = { index1, index2 };
				ptableColumns.forEach(col => {
					row[col] = jsonData[col][index]; // 修正箇所
				});
				return row;
			});
			ptable = response.data;

		} catch (err) {
			if (err.response) {
				ptableError = `確率表取得エラー: ${err.response.status} - ${err.response.statusText}`;
			} else if (err.request) {
				ptableError = '確率表取得エラー: サーバーからの応答がありません。';
			} else {
				ptableError = '確率表取得エラー: リクエストを送信できませんでした。';
			}
		} finally {
			ptableLoading = false;
		}
	}

	// 初期データ取得
	onMount(async () => {
		await Promise.all([fetchData(), fetchPtable()]);
	});

	// 都道府県変更時の処理
	function handlePrefectureChange(event) {
		prefecture = event.target.value;
		fetchData();
	}

	// 時刻変更時の処理
	function handleDateTimeChange(event) {
		selectedDateTime = event.target.value;
		fetchData();
	}

	// ptable から確率を取得する関数
	function getProbability(x, y) {
		if (!ptable) return null; // ptable が null の場合は null を返す
		if (!ptableRows || ptableRows.length === 0) return null;
		const row = ptableRows.find(row => Math.floor(x / 5) * 5 === row.index1 && row.index2 === y); // 修正箇所
		if (!row || row[zValue] === undefined) return null; // 修正箇所
		return Math.round(row[zValue] * 100);
	}

	// 確率に応じた背景色を返す関数
	function getBackgroundColor(ptable, oxData, hour, i) {
		if (!ptable) return 'white'; // ptable が null の場合は white を返す
		const probability = getProbability(Math.round(oxData[`+${hour}`][i]), hour);

		if (probability === null) return 'white'; // null の場合は white を返す

		let hue, saturation, lightness;

		if (probability <= 25) {
			// 0% - 25%: 水色から灰色へ
			hue = 180; // 水色
			saturation = 100 - (probability / 25) * 100; // 100% -> 0%
			lightness = 80 - (probability / 25) * 30; // 80% -> 50%
		} else if (probability <= 50) {
			// 25% - 50%: 灰色から赤色へ
			hue = 0; // 赤色
			saturation = ( (probability - 25) / 25) * 100; // 0% -> 100%
			lightness = 50 + ((probability - 25) / 25) * 0; // 50% -> 50%
		} else {
			// 50% 以上: 赤色
			hue = 0;
			saturation = 100;
			lightness = 50;
		}

		return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
	}
</script>

<main>
	<h1>OX 予測データ</h1>

	<div>
		<label for="prefecture">都道府県:</label>
		<select id="prefecture" bind:value={prefecture} on:change={handlePrefectureChange}>
			<option value="kanagawa">神奈川県</option>
			<!-- 他の都道府県もここに追加 -->
		</select>
	</div>

	<div>
		<label for="dateTime">時刻:</label>
		<select id="dateTime" bind:value={selectedDateTime} on:change={handleDateTimeChange}>
			<option value="now">現在時刻</option>
			<option value="2015-07-27T06:00:00+09:00">2015年7月27日午前6時</option>
			<option value="2011-08-13T06:00:00+09:00">2011年8月13日午前6時</option>
			<!-- 他の過去の時刻もここに追加 -->
		</select>
	</div>

	{#if loading}
		<p>データを読み込み中です...</p>
	{:else if error}
		<p style="color: red;">エラー: {error}</p>
	{:else}
		{#if oxData && oxData.XY && oxData.lon && oxData.lat && Array.isArray(oxData.XY) && Array.isArray(oxData.lon) && Array.isArray(oxData.lat)}
			<table>
				<thead>
					<tr>
						<th>XY</th>
						<th>経度</th>
						<th>緯度</th>
						{#each Array.from({ length: 24 }, (_, i) => i + 1) as hour}
							<th>+{hour}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each oxData.XY as item, i}
						<tr>
							<td>{oxData.XY[i]}</td>
							<td>{oxData.lon[i].toFixed(2)}</td>
							<td>{oxData.lat[i].toFixed(2)}</td>
							{#each Array.from({ length: 24 }, (_, j) => j + 1) as hour}
								<td
									class:highlight={Math.round(oxData[`+${hour}`][i]) >= 120}
									style="background-color: {getBackgroundColor(ptable, oxData, hour, i)};"
								>
									{Math.round(oxData[`+${hour}`][i])}
									{#if ptable}
										<br />
										{#if getProbability(Math.round(oxData[`+${hour}`][i]), hour) !== null}
											{getProbability(Math.round(oxData[`+${hour}`][i]), hour)}%
										{:else}
											-
										{/if}
									{/if}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p>データがありません。</p>
		{/if}
	{/if}

	{#if ptableLoading}
		<p>確率表を読み込み中です...</p>
	{:else if ptableError}
		<p style="color: red;">確率表エラー: {ptableError}</p>
	{:else if ptable}
		<h2>OX 予測値と実測値の関係</h2>
		<table>
			<thead>
				<tr>
					<th>予測値範囲</th>
					<th>予測時間</th>
					{#each ptableColumns as column}
						<th>{column} 以上</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each ptableRows as row}
					<tr>
						<td>{row.index1}〜{row.index1 + 5}</td>
						<td>{row.index2} 時間後</td>
						{#each ptableColumns as column}
							<td>{Math.round(row[column] * 100)}%</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 100%;
		margin: 0 auto;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 1em;
	}

	th,
	td {
		border: 1px solid #ccc;
		padding: 0.5em;
		text-align: center;
	}
	.highlight {
		color: green;
		font-weight: bold;
	}
</style>
