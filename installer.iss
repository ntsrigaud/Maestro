[Setup]
AppName=Maestro
AppVersion=0.1.0
DefaultDirName={autopf}\Maestro
DefaultGroupName=Maestro
UninstallDisplayIcon={app}\Maestro.exe
Compression=lzma2
SolidCompression=yes
OutputDir=dist
OutputBaseFilename=Maestro_Setup
SetupIconFile=docs\assets\icon.ico

[Files]
Source: "dist\Maestro\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Maestro"; Filename: "{app}\Maestro.exe"
Name: "{autodesktop}\Maestro"; Filename: "{app}\Maestro.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
Filename: "{app}\Maestro.exe"; Description: "{cm:LaunchProgram,Maestro}"; Flags: nowait postinstall skipifsilent
