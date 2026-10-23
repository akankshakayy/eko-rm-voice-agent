package co.eko.sahayak.recsync

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.work.WorkInfo
import androidx.work.WorkManager
import java.text.DateFormat
import java.util.Date
import java.util.concurrent.Executors

/** The one screen: setup (server, token, Eko SIM, recordings folder) and the last run's status. */
class MainActivity : Activity() {
    private lateinit var s: Settings
    private var sims: List<Device.Sim> = emptyList()
    private val io = Executors.newSingleThreadExecutor()

    private val server by lazy { findViewById<EditText>(R.id.server) }
    private val token by lazy { findViewById<EditText>(R.id.token) }
    private val testResult by lazy { findViewById<TextView>(R.id.test_result) }
    private val simSpinner by lazy { findViewById<Spinner>(R.id.sim) }
    private val folder by lazy { findViewById<TextView>(R.id.folder) }
    private val wifiOnly by lazy { findViewById<CheckBox>(R.id.wifi_only) }
    private val status by lazy { findViewById<TextView>(R.id.status) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        s = Settings(this)
        server.setText(s.serverUrl)
        token.setText(s.token)
        wifiOnly.isChecked = s.wifiOnly
        showFolder()

        findViewById<Button>(R.id.test).setOnClickListener { testConnection() }
        findViewById<Button>(R.id.pick_folder).setOnClickListener {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_FOLDER)
        }
        findViewById<Button>(R.id.save).setOnClickListener { save() }
        findViewById<Button>(R.id.upload_now).setOnClickListener {
            if (!s.isConfigured) return@setOnClickListener toast("Finish setup and tap Save first")
            UploadWorker.runNow(this)
            toast("Uploading in the background")
        }
        WorkManager.getInstance(this).getWorkInfosForUniqueWorkLiveData("upload-now").observeForever { infos ->
            if (infos.any { it.state.isFinished || it.state == WorkInfo.State.ENQUEUED }) showStatus()
        }

        val missing = PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) loadSims() else requestPermissions(missing.toTypedArray(), ASK_PERMISSIONS)
    }

    override fun onResume() {
        super.onResume()
        showStatus()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grants: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grants)
        if (grants.all { it == PackageManager.PERMISSION_GRANTED }) loadSims()
        else testResult.text = "Call log and phone permissions are needed to tell Eko SIM calls apart. " +
            "If Android says \"Restricted setting\", open App info › ⋮ › Allow restricted settings, then reopen the app."
    }

    private fun loadSims() {
        sims = try { Device.sims(this) } catch (e: SecurityException) { emptyList() }
        simSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            sims.map { it.label }.ifEmpty { listOf("No SIM found") })
        sims.indexOfFirst { it.subId == s.ekoSubId }.takeIf { it >= 0 }?.let(simSpinner::setSelection)
    }

    @Deprecated("Activity result API needs androidx.activity; the framework callback keeps the app dependency-light")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode != PICK_FOLDER || resultCode != RESULT_OK || uri == null) return
        contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        s.folderUri = uri.toString()
        showFolder()
    }

    private fun testConnection() {
        val url = server.text.toString().trim().trimEnd('/')
        Matcher.serverUrlProblem(url)?.let { testResult.text = it; return }
        testResult.text = "Checking…"
        val tok = token.text.toString().trim()
        io.execute {
            val msg = try {
                val name = Api(url, tok).ping()
                s.rmName = name
                "Connected. Uploads will be filed under RM: $name"
            } catch (e: Exception) {
                "Couldn't connect: ${e.message}"
            }
            runOnUiThread { testResult.text = msg }
        }
    }

    private fun save() {
        val url = server.text.toString().trim().trimEnd('/')
        Matcher.serverUrlProblem(url)?.let { return toast(it) }
        if (token.text.isBlank()) return toast("Paste the upload token")
        val sim = sims.getOrNull(simSpinner.selectedItemPosition) ?: return toast("Choose the Eko SIM")
        if (s.folderUri.isEmpty()) return toast("Choose the call recordings folder")
        s.serverUrl = url
        s.token = token.text.toString()
        s.ekoSubId = sim.subId
        s.ekoSimLabel = sim.label
        s.ekoAccountIds = sim.accountIds
        s.wifiOnly = wifiOnly.isChecked
        UploadWorker.scheduleDaily(this)
        UploadWorker.watchCalls(this)
        toast("Saved. Recordings upload every day at about 8 pm.")
        showStatus()
    }

    private fun showFolder() {
        folder.text = if (s.folderUri.isEmpty()) "No folder chosen" else
            DocumentsContract.getTreeDocumentId(Uri.parse(s.folderUri)).replace("primary:", "Internal storage/")
    }

    private fun showStatus() {
        val st = s.status()
        val fmt = DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT)
        status.text = buildString {
            appendLine(if (s.isConfigured) "Daily upload is on (Eko SIM: ${s.ekoSimLabel})" else "Setup not finished")
            if (s.rmName.isNotEmpty()) appendLine("RM: ${s.rmName}")
            if (st.at == 0L) appendLine("No upload run yet") else {
                appendLine("Last run: ${fmt.format(Date(st.at))}")
                appendLine("Uploaded ${st.uploaded} · already on server ${st.duplicates} · " +
                    "skipped ${st.skipped} (not Eko SIM / no call) · failed ${st.failed}")
            }
            st.error?.let { appendLine("Last problem: $it") }
            append("Uploading recordings made since ${fmt.format(Date(s.watermarkMs))}")
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    companion object {
        private const val PICK_FOLDER = 1
        private const val ASK_PERMISSIONS = 2
        val PERMISSIONS = listOf(Manifest.permission.READ_CALL_LOG, Manifest.permission.READ_PHONE_STATE)
    }
}
