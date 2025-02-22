<?php
require 'C:\xampp\htdocs\tesis\vendor\autoload.php'; // Asegúrate de tener PhpSpreadsheet instalado

use PhpOffice\PhpSpreadsheet\IOFactory;

$filePath = 'C:\xampp\htdocs\tesis\dataframe\datosbase\conjunto_prueba.xlsx';

if (file_exists($filePath)) {
    try {
        $spreadsheet = IOFactory::load($filePath);
        $worksheet = $spreadsheet->getActiveSheet();
        $data = [];

        // Omitir la primera fila (encabezados)
        foreach ($worksheet->getRowIterator() as $rowIndex => $row) {
            if ($rowIndex > 1) { // Comienza desde la segunda fila
                $rowData = [];
                foreach ($row->getCellIterator() as $cell) {
                    if ($cell->getValue() !== null) { 
                        $rowData[] = (string)$cell->getValue(); 
                    }
                }
                if (!empty($rowData)) { 
                    $data[] = array_map('strval', $rowData); 
                }
            }
        }

        echo json_encode(['status' => 'ok', 'data' => $data]);
    } catch (Exception $e) {
        echo json_encode(['status' => 'error', 'message' => 'Ocurrió un error al leer el archivo: ' . $e->getMessage()]);
    }
} else {
    echo json_encode(['status' => 'error', 'message' => 'El archivo no existe']);
}
?>

